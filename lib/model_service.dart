import 'package:onnxruntime/onnxruntime.dart' as ort;
import 'package:flutter/services.dart' show rootBundle;
import 'dart:math';
import 'dart:typed_data';

class ModelService {
  ort.OrtSession? _fruitSession; // For model.onnx (fruit)
  ort.OrtSession? _leafSession; // For LeafModel.onnx (leaf)
  ort.OrtSessionOptions? _sessionOptions;
  ort.OrtRunOptions? _runOptions;

  final List<String> fruitLabels = [
    "Bacterial Blight",
    "Calyx Rot",
    "Fungal Cercospora",
    "Fruit Rot",
    "Healthy",
    "Fungal Scab",
  ];

  final List<String> leafLabels = ["Bacterial", "Fungal", "Healthy"];

  Future<void> loadModels() async {
    try {
      // Initialize ONNX Runtime environment
      ort.OrtEnv.instance.init();

      // Load session options
      _sessionOptions = ort.OrtSessionOptions();

      // Load the Fruit model (model.onnx)
      final fruitRawAssetFile = await rootBundle.load("assets/CoAtNet.onnx");
      final fruitBytes = fruitRawAssetFile.buffer.asUint8List();
      _fruitSession = ort.OrtSession.fromBuffer(fruitBytes, _sessionOptions!);

      // Load the Leaf model (LeafModel.onnx)
      final leafRawAssetFile = await rootBundle.load("assets/LeafModel.onnx");
      final leafBytes = leafRawAssetFile.buffer.asUint8List();
      _leafSession = ort.OrtSession.fromBuffer(leafBytes, _sessionOptions!);

      // Initialize run options
      _runOptions = ort.OrtRunOptions();

      print('✅ Both ONNX Models loaded successfully');
    } catch (e) {
      print('❌ Error loading ONNX models: $e');
      rethrow; // Rethrow to ensure caller knows loading failed
    }
  }

  Future<Map<String, dynamic>> runInference(
    Float32List inputData,
    String modelType,
  ) async {
    if (_fruitSession == null || _leafSession == null || _runOptions == null) {
      throw Exception("Models not loaded yet");
    }

    // Create tensor with preprocessed image data
    final inputOrt = ort.OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, 224, 224], // Shape: (1, 3, 224, 224)
    );

    // Define inputs (assuming both models expect an input named "input")
    final inputs = {'input': inputOrt};

    // Run inference with the selected model
    final outputs = (modelType == 'Fruit' ? _fruitSession! : _leafSession!).run(
      _runOptions!,
      inputs,
    );

    // Process output
    final outputTensor = outputs[0]?.value;
    final List<double> probabilities;
    if (outputTensor is List && outputTensor.isNotEmpty) {
      // The output shape is typically [1, num_classes], which the onnxruntime
      // package represents as a List<List<double>>. We extract the inner list.
      if (outputTensor.first is List) {
        probabilities = List<double>.from(outputTensor.first as List);
      } else {
        // Fallback for a flat list output, shape [num_classes]
        probabilities = List<double>.from(outputTensor);
      }
    } else {
      throw Exception("Unexpected output format: $outputTensor");
    }

    // Get predicted class from the probabilities
    final predictedClass = probabilities.indexOf(
      probabilities.reduce((a, b) => a > b ? a : b),
    );

    // Clean up input tensor
    inputOrt.release();

    return {'class': predictedClass, 'probabilities': probabilities};
  }

  String getLabel(int classIndex, String modelType) {
    return modelType == 'Fruit'
        ? fruitLabels[classIndex]
        : leafLabels[classIndex];
  }

  void dispose() {
    _runOptions?.release();
    _fruitSession?.release();
    _leafSession?.release();
    _sessionOptions?.release();
    ort.OrtEnv.instance.release();
    print('✅ ONNX Models released');
  }
}
