import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'image_processor.dart';
import 'model_service.dart';
import 'package:csv/csv.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';


class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? filePath;
  String label = "Label";
  late ModelService _modelService;
  late ImageProcessor _imageProcessor;
  String _selectedModel = 'Fruit'; // Default to Fruit Model
  bool _isLoadingModels = true; // Track model loading state
  bool _isProcessingImage = false; // Track inference state
  bool _isBatchProcessing = false; // Track batch processing state
  final List<List<dynamic>> _csvData = [];

  @override
  void initState() {
    super.initState();
    _modelService = ModelService();
    _imageProcessor = ImageProcessor(_modelService);
    _loadModels();
    // Add CSV header row
    _csvData.add([
      'image',
      'predicted_class',
      ..._modelService.fruitLabels, // Assuming fruit model for headers
    ]);
  }

  Future<void> _loadModels() async {
    await _modelService.loadModels();
    setState(() {
      _isLoadingModels = false; // Models are loaded, enable UI
    });
  }

  Future<void> _pickAndProcessImage(ImageSource source) async {
    if (_isLoadingModels || _isProcessingImage) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text(_isLoadingModels
                ? "Please wait, models are loading..."
                : "Already processing an image.")),
      );
      return;
    }

    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: source);
    if (image == null) return;

    File? croppedImage = await _imageProcessor.cropImage(File(image.path));
    if (croppedImage == null) return;

    // Show the image and a loading indicator immediately
    setState(() {
      filePath = croppedImage;
      _isProcessingImage = true;
      label = "Analyzing...";
    });

    // Run inference in the background
    final result =
        await _imageProcessor.runInference(croppedImage, _selectedModel);

    // Update UI with the result
    if (mounted) {
      setState(() {
        if (result != null) {
          label = result['label'] as String;
          // Store data for CSV export, only for the Fruit model as requested
          if (_selectedModel == 'Fruit') {
            _csvData.add([
              result['imageName'],
              result['label'],
              ...(result['probabilities'] as List<double>)
                  .map((p) => p.toStringAsFixed(3)) // Format to 3 decimal places
                  .toList(),
            ]);
          }
        } else {
          label = "Error during analysis";
        }
        _isProcessingImage = false;
      });
    }
  }

  Future<void> _runBatchTest() async {
    if (_isProcessingImage || _isBatchProcessing) return;

    setState(() {
      _isBatchProcessing = true;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
          content:
              Text("Running batch test on all images in assets/testing...")),
    );

    final List<List<dynamic>>? batchCsvData =
        await _imageProcessor.runBatchInference(_selectedModel);

    if (mounted) {
      setState(() {
        _isBatchProcessing = false;
      });
    }

    if (batchCsvData != null) {
      await _shareCsv(batchCsvData, "pomo_ai_batch_predictions.csv");
    } else if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text("Batch test failed or no images found.")),
      );
    }
  }

  Future<void> _exportAndShareCsv() async {
    await _shareCsv(_csvData, "pomo_ai_predictions.csv");
  }

  Future<void> _shareCsv(List<List<dynamic>> data, String fileName) async {
    if (data.length <= 1) { // Only header exists
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No data to export.")),
      );
      return;
    }
    final String csvString = const ListToCsvConverter().convert(data);
    final Directory directory = await getApplicationDocumentsDirectory();
    final String path = '${directory.path}/$fileName';
    final File file = File(path);
    await file.writeAsString(csvString);

    Share.shareXFiles([XFile(path)], text: 'Here are the Pomo AI predictions!');
  }

  @override
  void dispose() {
    _modelService.dispose(); // Release models when the page is closed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Pomo AI")),
      body:
          _isLoadingModels
              ? const Center(
                child: CircularProgressIndicator(),
              ) // Show loading indicator
              : SingleChildScrollView(
                child: Center(
                  child: Column(
                    children: [
                      const SizedBox(height: 12),
                      Card(
                        elevation: 20,
                        clipBehavior: Clip.hardEdge,
                        child: SizedBox(
                          width: 300,
                          child: Column(
                            children: [
                              const SizedBox(height: 18),
                              Container(
                                height: 280,
                                width: 280,
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(12),
                                  image:
                                      filePath == null
                                          ? const DecorationImage(
                                            image: AssetImage(
                                              'assets/upload.png',
                                            ),
                                            fit: BoxFit.cover,
                                          )
                                          : null,
                                ),
                                child:
                                    filePath == null
                                        ? null
                                        : ClipRRect(
                                          borderRadius: BorderRadius.circular(
                                            12,
                                          ),
                                          child: Image.file(
                                            filePath!,
                                            fit: BoxFit.cover,
                                          ),
                                        ),
                              ),
                              const SizedBox(height: 12),
                              Padding(
                                padding: const EdgeInsets.all(8.0),
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    if (_isProcessingImage) ...[
                                      const SizedBox(
                                          height: 20,
                                          width: 20,
                                          child: CircularProgressIndicator()),
                                      const SizedBox(width: 10),
                                    ],
                                    Text(label,
                                        style: const TextStyle(
                                            fontSize: 18, fontWeight: FontWeight.bold)),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      // Model selection buttons
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          ElevatedButton(
                            onPressed: _isProcessingImage || _isBatchProcessing
                                ? null
                                : () {
                              setState(() {
                                _selectedModel = 'Fruit';
                                label = "Label"; // Reset label when switching
                              });
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor:
                                  _selectedModel == 'Fruit'
                                      ? Colors.deepPurple
                                      : null,
                            ),
                            child: const Text("Fruit Model"),
                          ),
                          const SizedBox(width: 12),
                          ElevatedButton(
                            onPressed: _isProcessingImage || _isBatchProcessing
                                ? null
                                : () {
                              setState(() {
                                _selectedModel = 'Leaf';
                                label = "Label"; // Reset label when switching
                              });
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor:
                                  _selectedModel == 'Leaf'
                                      ? Colors.deepPurple
                                      : null,
                            ),
                            child: const Text("Leaf Model"),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          ElevatedButton(
                            onPressed: _isProcessingImage || _isBatchProcessing ? null : () =>
                                _pickAndProcessImage(ImageSource.gallery),
                            child: const Text("Gallery"),
                          ),
                          const SizedBox(width: 12),
                          ElevatedButton(
                            onPressed: _isProcessingImage || _isBatchProcessing ? null : () =>
                                _pickAndProcessImage(ImageSource.camera),
                            child: const Text("Camera"),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          ElevatedButton(
                            onPressed: _isProcessingImage || _isBatchProcessing ? null : _exportAndShareCsv,
                            child: const Text("Export CSV"),
                          ),
                          const SizedBox(width: 12),
                          ElevatedButton(
                            onPressed: _isProcessingImage || _isBatchProcessing ? null : _runBatchTest,
                            child: const Text("Run Batch Test"),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
    );
  }
}
