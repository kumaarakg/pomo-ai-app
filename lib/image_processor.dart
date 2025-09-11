import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image/image.dart' as img;
import 'model_service.dart';

class ImageProcessor {
  final ModelService _modelService;

  ImageProcessor(this._modelService);

  Future<File?> cropImage(File imageFile) async {
    CroppedFile? croppedFile = await ImageCropper().cropImage(
      sourcePath: imageFile.path,
      aspectRatio: const CropAspectRatio(ratioX: 1, ratioY: 1),
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Crop Image',
          toolbarColor: Colors.deepPurple,
          toolbarWidgetColor: Colors.white,
          lockAspectRatio: false,
        ),
        IOSUiSettings(title: 'Crop Image', minimumAspectRatio: 1.0),
      ],
    );

    return croppedFile != null ? File(croppedFile.path) : null;
  }

  Future<Map<String, dynamic>?> runInference(
    File imageFile,
    String modelType, // "Fruit" or "Leaf"
  ) async {
    try {
      // Run image processing in a separate isolate to avoid blocking the UI
      final preprocessedResult = await compute(_preprocessImage, imageFile);
      final inputData = preprocessedResult['inputData'] as Float32List;
      final imageName = preprocessedResult['imageName'] as String;

      // Run inference and get results
      final result = await _modelService.runInference(inputData, modelType);
      final newLabel = _modelService.getLabel(result['class'], modelType);
      final probabilities = result['probabilities'] as List<double>;

      print(
        "✅ Prediction: $newLabel",
      );
      return {'label': newLabel, 'probabilities': probabilities, 'imageName': imageName};
    } catch (e) {
      print("❌ Error running inference: $e");
      return null;
    }
  }

  Future<List<List<dynamic>>?> runBatchInference(String modelType) async {
    try {
      // 1. Load the AssetManifest to find all assets
      final manifestContent = await rootBundle.loadString('AssetManifest.json');
      final Map<String, dynamic> manifestMap = json.decode(manifestContent);

      // 2. Filter for images in the assets/testing/ directory
      final imagePaths = manifestMap.keys
          .where((String key) => key.startsWith('assets/testing/'))
          .toList();

      if (imagePaths.isEmpty) {
        print("⚠️ No images found in assets/testing/");
        return null;
      }

      print("Found ${imagePaths.length} images for batch testing.");

      // 3. Prepare CSV data structure
      final List<String> labels = (modelType == 'Fruit')
          ? _modelService.fruitLabels
          : _modelService.leafLabels;
      final List<List<dynamic>> csvData = [
        ['image', 'predicted_class', ...labels]
      ];

      // 4. Loop through each image, process, and run inference
      for (final path in imagePaths) {
        print("Processing $path...");
        final byteData = await rootBundle.load(path);

        // Preprocess the image in an isolate
        final inputData =
            await compute(_preprocessImageAsset, byteData.buffer.asUint8List());

        // Run inference
        final result = await _modelService.runInference(inputData, modelType);
        final predictedLabel = _modelService.getLabel(result['class'], modelType);
        final probabilities = result['probabilities'] as List<double>;
        final imageName = path.split('/').last;

        // Add row to CSV data
        csvData.add([
          imageName,
          predictedLabel,
          ...probabilities.map((p) => p.toStringAsFixed(3)).toList(),
        ]);
      }

      print("✅ Batch processing complete.");
      return csvData;
    } catch (e) {
      print("❌ Error during batch inference: $e");
      return null;
    }
  }
}

// This top-level function will run in a separate isolate.
Map<String, dynamic> _preprocessImage(File imageFile) {
  // Decode image using synchronous method as it's in a background isolate
  img.Image? image = img.decodeImage(imageFile.readAsBytesSync());
  if (image == null) {
    throw Exception("Error decoding image");
  }

  // Resize to 224x224
  image = img.copyResize(image, width: 224, height: 224);

  // Convert to (1, 3, 224, 224) tensor (RGB order)
  final inputData = Float32List(1 * 3 * 224 * 224);
  int pixelIndex = 0;
  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      final pixel = image.getPixel(x, y);
      inputData[pixelIndex] = pixel.r.toDouble(); // R channel
      inputData[pixelIndex + 224 * 224] = pixel.g.toDouble(); // G channel
      inputData[pixelIndex + 2 * 224 * 224] = pixel.b.toDouble(); // B channel
      pixelIndex++;
    }
  }

  return {
    'inputData': inputData,
    'imageName': imageFile.path.split('/').last,
  };
}

// This top-level function will run in a separate isolate for batch processing.
Float32List _preprocessImageAsset(Uint8List imageBytes) {
  // Decode image using synchronous method as it's in a background isolate
  img.Image? image = img.decodeImage(imageBytes);
  if (image == null) {
    throw Exception("Error decoding image");
  }

  // Resize to 224x224
  image = img.copyResize(image, width: 224, height: 224);

  // Convert to (1, 3, 224, 224) tensor (RGB order)
  final inputData = Float32List(1 * 3 * 224 * 224);
  int pixelIndex = 0;
  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      final pixel = image.getPixel(x, y);
      inputData[pixelIndex] = pixel.r.toDouble(); // R channel
      inputData[pixelIndex + 224 * 224] = pixel.g.toDouble(); // G channel
      inputData[pixelIndex + 2 * 224 * 224] = pixel.b.toDouble(); // B channel
      pixelIndex++;
    }
  }

  return inputData;
}
