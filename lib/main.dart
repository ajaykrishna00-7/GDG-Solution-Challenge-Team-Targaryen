// ignore_for_file: deprecated_member_use, library_private_types_in_public_api

// Import necessary Flutter and third-party packages
import 'package:flutter/material.dart';
import 'package:confetti/confetti.dart'; // For celebratory animations
import 'package:google_fonts/google_fonts.dart'; // For custom fonts
import 'dart:convert'; // For JSON parsing
import 'package:flutter/services.dart' show rootBundle; // For loading assets
import 'dart:math'; // For math operations and random number generation

// Entry point of the application
void main() {
  runApp(const ChemMixApp());
}

/**
 * MLModel class for handling the machine learning model operations
 * This class loads the trained model, preprocesses inputs, and makes predictions
 */
class MLModel {
  // Model parameters
  late List<List<double>> coef; // Coefficients matrix
  late List<double> intercept; // Intercept values
  late List<double> mean; // Mean values for feature scaling
  late List<double> scale; // Scale values for feature scaling

  // Helper method to load JSON data from assets
  Future<Map<String, dynamic>> loadJsonFile(String path) async {
    String jsonString = await rootBundle.loadString(path);
    return jsonDecode(jsonString);
  }

  // Loads model and scaler parameters from JSON files
  Future<void> loadModel() async {
    String modelJson = await rootBundle.loadString('assets/model.json');
    String scalerJson = await rootBundle.loadString('assets/scaler.json');
    final modelData = jsonDecode(modelJson);
    final scalerData = jsonDecode(scalerJson);

    // Parse coefficients array from model data
    coef = (modelData["coef"] as List)
        .map((row) => (row as List)
            .map((e) => double.tryParse(e.toString()) ?? (e as num).toDouble())
            .toList())
        .toList();

    // Parse intercept array from model data
    intercept = (modelData["intercept"] as List)
        .map((e) => double.tryParse(e.toString()) ?? (e as num).toDouble())
        .toList();

    // Parse mean array from scaler data (for standardization)
    mean = (scalerData["mean"] as List)
        .map((e) => double.tryParse(e.toString()) ?? (e as num).toDouble())
        .toList();

    // Parse scale array from scaler data (for standardization)
    scale = (scalerData["scale"] as List)
        .map((e) => double.tryParse(e.toString()) ?? (e as num).toDouble())
        .toList();
  }

  // Apply standardization to input features using loaded mean and scale
  List<double> scaleInput(List<double> input) {
    return List.generate(input.length, (i) => (input[i] - mean[i]) / scale[i]);
  }

  // Make prediction using the loaded model
  // Returns probability score for binary classification
  double predict(List<double> input) {
    List<double> scaledInput = scaleInput(input);
    double prediction = 0.0;

    // Apply model coefficients and intercept
    for (int i = 0; i < coef.length; i++) {
      double sum = 0.0;
      for (int j = 0; j < scaledInput.length; j++) {
        sum += coef[i][j] * scaledInput[j];
      }
      prediction = sum + intercept[i];
    }

    // Apply sigmoid function to get probability (0-1 range)
    return 1 / (1 + exp(-prediction));
  }
}

/**
 * Main app widget that defines the theme and global styling
 */
class ChemMixApp extends StatelessWidget {
  const ChemMixApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ChemPredict Pro',
      debugShowCheckedModeBanner: false, // Remove debug banner
      theme: ThemeData(
        // Dark blue color scheme
        primaryColor: const Color(0xFF0A192F), // Navy blue
        colorScheme: ColorScheme.light(
          primary: const Color(0xFF0A192F),
          secondary: const Color(0xFF64FFDA), // Teal
          surface: const Color(0xFF172A45),
        ),
        scaffoldBackgroundColor: const Color(0xFF0A192F),
        // Typography with Roboto font
        textTheme: GoogleFonts.robotoTextTheme().copyWith(
          displayLarge: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
          bodyLarge: const TextStyle(color: Colors.white70),
          bodyMedium: const TextStyle(color: Colors.white70),
        ),
        // Card styling with rounded corners
        cardTheme: CardTheme(
          elevation: 8,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          color: const Color(0xFF172A45),
        ),
        // Input field styling
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: const Color(0xFF172A45).withOpacity(0.5),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide.none,
          ),
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 20,
            vertical: 18,
          ),
        ),
      ),
      home: const HomePage(), // Set the home page
    );
  }
}

/**
 * HomePage is the main screen of the application with multiple tabs
 */
class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage>
    with SingleTickerProviderStateMixin {
  // Controllers
  late TabController _tabController; // For managing tabs
  final TextEditingController smilesController =
      TextEditingController(); // For SMILES input
  final TextEditingController nameController =
      TextEditingController(); // For chemical name input
  bool isPredicting = false; // Tracks prediction state
  bool showResult = false; // Controls when to show results
  late ConfettiController _confettiController; // For celebration animation
  final MLModel mlModel = MLModel(); // ML model instance

  // Structure to store prediction results
  Map<String, dynamic> predictionResults = {
    'name': '',
    'smiles': '',
    'properties': {
      'molecular_weight': 0.0,
      'mol_logp': 0.0,
      'tpsa': 0.0,
      'h_donors': 0,
      'h_acceptors': 0,
      'ring_count': 0,
      'rotatable_bonds': 0,
      'fraction_csp3': 0.0,
      'toxicity': 0,
      'toxicity_confidence': 0.0,
    }
  };

  @override
  void initState() {
    super.initState();
    // Initialize tab controller with 4 tabs
    _tabController = TabController(length: 4, vsync: this);
    // Set up confetti controller for 5-second duration
    _confettiController =
        ConfettiController(duration: const Duration(seconds: 5));
    // Load ML model on startup
    mlModel.loadModel();
  }

  @override
  void dispose() {
    // Clean up controllers when widget is removed
    _tabController.dispose();
    smilesController.dispose();
    nameController.dispose();
    _confettiController.dispose();
    super.dispose();
  }

  // Method to predict chemical properties using ML model
  Future<void> predict() async {
    // Validate input fields
    if (smilesController.text.trim().isEmpty ||
        nameController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text(
              'Please fill in both SMILES notation and chemical name'),
          backgroundColor: Colors.red[800],
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      );
      return;
    }

    // Update UI to show prediction in progress
    setState(() {
      isPredicting = true;
      showResult = false;
    });

    // Generate random molecular descriptors (would be calculated from SMILES in a real app)
    List<double> descriptors =
        List.generate(8, (index) => Random().nextDouble());

    // Load model and make prediction
    MLModel model = MLModel();
    await model.loadModel();
    double toxicityPrediction = model.predict(descriptors);
    int toxicity = toxicityPrediction > 0.5 ? 1 : 0; // Binary classification

    // Update UI with prediction results
    setState(() {
      isPredicting = false;
      showResult = true;
      predictionResults = {
        'name': nameController.text,
        'smiles': smilesController.text,
        'properties': {
          // Scale random values to realistic property ranges
          'molecular_weight':
              descriptors[0] * 300 + 50, // Scale to realistic values
          'mol_logp': descriptors[1] * 10 - 5,
          'tpsa': descriptors[2] * 150,
          'h_donors': (descriptors[3] * 5).round(),
          'h_acceptors': (descriptors[4] * 10).round(),
          'ring_count': (descriptors[5] * 4).round(),
          'rotatable_bonds': (descriptors[6] * 8).round(),
          'fraction_csp3': descriptors[7],
          'toxicity': toxicity,
          'toxicity_confidence': toxicityPrediction,
        }
      };
      // Navigate to results tab and play confetti animation
      _tabController.animateTo(1);
      _confettiController.play();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Scaffold(
          backgroundColor: Colors.transparent,
          appBar: AppBar(
            // Application header with logo and title
            title: Row(
              children: [
                const Icon(Icons.science, size: 30, color: Color(0xFF64FFDA)),
                const SizedBox(width: 12),
                Text(
                  'ChemPredict Pro',
                  style: GoogleFonts.orbitron(
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.5,
                  ),
                ),
              ],
            ),
            centerTitle: false,
            elevation: 0,
            backgroundColor: Colors.transparent,
            // Tab bar with 4 sections
            bottom: TabBar(
              controller: _tabController,
              indicatorColor: const Color(0xFF64FFDA),
              indicatorWeight: 3,
              labelStyle: GoogleFonts.roboto(
                fontWeight: FontWeight.w500,
                fontSize: 16,
              ),
              unselectedLabelColor: Colors.black54,
              tabs: const [
                Tab(icon: Icon(Icons.science), text: 'Predict'),
                Tab(icon: Icon(Icons.analytics), text: 'Results'),
                Tab(icon: Icon(Icons.insights), text: 'Analysis'),
                Tab(icon: Icon(Icons.people), text: 'Team'),
              ],
            ),
            actions: [
              // Theme toggle button (functionality not implemented)
              IconButton(
                icon: const Icon(Icons.brightness_4),
                onPressed: () {},
                tooltip: 'Toggle Theme',
              ),
              const SizedBox(width: 16),
            ],
          ),
          // Main content area with tab views
          body: TabBarView(
            controller: _tabController,
            children: [
              buildPredictTab(),
              buildResultsTab(),
              buildAnalysisTab(),
              buildTeamTab(),
            ],
          ),
        ),
        // Confetti animation overlay for successful predictions
        Align(
          alignment: Alignment.topCenter,
          child: ConfettiWidget(
            confettiController: _confettiController,
            blastDirectionality: BlastDirectionality.explosive,
            shouldLoop: false,
            colors: const [
              Color(0xFF64FFDA),
              Colors.white,
              Color(0xFF0A192F),
            ],
          ),
        ),
      ],
    );
  }

  // First tab: Input chemical data for prediction
  Widget buildPredictTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Tab header
          Text(
            'Molecular Prediction Engine',
            style: GoogleFonts.orbitron(
              fontSize: 28,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF64FFDA),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Enter chemical details to predict properties and interactions',
            style: GoogleFonts.roboto(
              fontSize: 16,
              color: Colors.white70,
            ),
          ),
          const SizedBox(height: 32),
          // Chemical input card
          Card(
            elevation: 12,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Chemical Input',
                    style: GoogleFonts.roboto(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 24),
                  // SMILES notation input field
                  TextField(
                    controller: smilesController,
                    decoration: InputDecoration(
                      labelText: 'SMILES Notation',
                      labelStyle: const TextStyle(color: Colors.white70),
                      prefixIcon:
                          const Icon(Icons.code, color: Color(0xFF64FFDA)),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      filled: true,
                      fillColor: const Color(0xFF172A45).withOpacity(0.7),
                    ),
                    style: GoogleFonts.robotoMono(
                      color: Colors.white,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 24),
                  // Chemical name input field
                  TextField(
                    controller: nameController,
                    decoration: InputDecoration(
                      labelText: 'Chemical Name',
                      labelStyle: const TextStyle(color: Colors.white70),
                      prefixIcon: const Icon(Icons.emoji_objects,
                          color: Color(0xFF64FFDA)),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      filled: true,
                      fillColor: const Color(0xFF172A45).withOpacity(0.7),
                    ),
                    style: GoogleFonts.roboto(
                      color: Colors.white,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 32),
                  // Prediction button
                  SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton(
                      onPressed: !isPredicting ? predict : null,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF64FFDA),
                        foregroundColor: const Color(0xFF0A192F),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 8,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          // Show spinner during prediction
                          if (isPredicting)
                            const SizedBox(
                              width: 24,
                              height: 24,
                              child: CircularProgressIndicator(
                                color: Color(0xFF0A192F),
                                strokeWidth: 3,
                              ),
                            )
                          else
                            const Icon(Icons.auto_awesome, size: 24),
                          const SizedBox(width: 12),
                          Text(
                            isPredicting
                                ? 'Analyzing Structure...'
                                : 'Predict Properties',
                            style: GoogleFonts.roboto(
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 32),
          // Example chemicals section
          Text(
            'Common Chemical Examples',
            style: GoogleFonts.roboto(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 16),
          // Chips for quick selection of common chemicals
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              'H2O (Water)',
              'NaCl (Sodium Chloride)',
              'HCl (Hydrochloric Acid)',
              'NaOH (Sodium Hydroxide)',
              'CH3COOH (Acetic Acid)',
              'C2H5OH (Ethanol)',
              'H2SO4 (Sulfuric Acid)',
            ].map((chem) {
              return ActionChip(
                label: Text(chem),
                backgroundColor: const Color(0xFF172A45),
                labelStyle: GoogleFonts.robotoMono(
                  color: const Color(0xFF64FFDA),
                ),
                onPressed: () {
                  // Extract chemical name from the display text
                  final parts = chem.split(' ');
                  nameController.text =
                      parts[1].replaceAll('(', '').replaceAll(')', '');
                  // Notify user to enter SMILES manually
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('Selected $chem - enter SMILES manually'),
                      backgroundColor: const Color(0xFF64FFDA),
                      behavior: SnackBarBehavior.floating,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  );
                },
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  // Second tab: Display prediction results
  Widget buildResultsTab() {
    // Show placeholder when no prediction has been made
    if (!showResult) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.science,
              size: 80,
              color: Colors.white.withOpacity(0.3),
            ),
            const SizedBox(height: 24),
            Text(
              'Enter chemical details to see predictions',
              style: GoogleFonts.roboto(
                fontSize: 18,
                color: Colors.white70,
              ),
            ),
          ],
        ),
      );
    }

    // Display prediction results
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Tab header
          Text(
            'Prediction Results',
            style: GoogleFonts.orbitron(
              fontSize: 28,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF64FFDA),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Analysis for ${predictionResults['name']}',
            style: GoogleFonts.roboto(
              fontSize: 18,
              color: Colors.white70,
            ),
          ),
          const SizedBox(height: 32),
          // Results card
          Card(
            elevation: 12,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Chemical header with first letter icon
                  Row(
                    children: [
                      Container(
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          color: const Color(0xFF172A45),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Center(
                          child: Text(
                            predictionResults['name']
                                .toString()
                                .substring(0, 1),
                            style: GoogleFonts.orbitron(
                              fontSize: 36,
                              fontWeight: FontWeight.bold,
                              color: const Color(0xFF64FFDA),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 24),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              predictionResults['name'],
                              style: GoogleFonts.roboto(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              predictionResults['smiles'],
                              style: GoogleFonts.robotoMono(
                                fontSize: 16,
                                color: Colors.white70,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 32),
                  // Safety assessment section
                  Text(
                    'Safety Assessment',
                    style: GoogleFonts.roboto(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 16),
                  // Safety status container with color coding
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: predictionResults['properties']['toxicity'] == 0
                          ? Colors.green.withOpacity(0.2)
                          : Colors.red.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: predictionResults['properties']['toxicity'] == 0
                            ? Colors.green
                            : Colors.red,
                        width: 1,
                      ),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          predictionResults['properties']['toxicity'] == 0
                              ? Icons.check_circle
                              : Icons.warning,
                          color:
                              predictionResults['properties']['toxicity'] == 0
                                  ? Colors.green
                                  : Colors.red,
                          size: 36,
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                predictionResults['properties']['toxicity'] == 0
                                    ? 'Moderately Safe'
                                    : 'Potentially Hazardous',
                                style: GoogleFonts.roboto(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                'Confidence: ${(predictionResults['properties']['toxicity_confidence'] * 100).toStringAsFixed(1)}%',
                                style: GoogleFonts.roboto(
                                  fontSize: 14,
                                  color: Colors.white70,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),
                  // Key properties section
                  Text(
                    'Key Properties',
                    style: GoogleFonts.roboto(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 16),
                  // Grid of property tiles
                  GridView.count(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    crossAxisCount: 2,
                    crossAxisSpacing: 16,
                    mainAxisSpacing: 16,
                    childAspectRatio: 3,
                    children: [
                      _buildPropertyTile('Molecular Weight',
                          '${predictionResults['properties']['molecular_weight'].toStringAsFixed(2)} g/mol'),
                      _buildPropertyTile(
                          'LogP',
                          predictionResults['properties']['mol_logp']
                              .toStringAsFixed(3)),
                      _buildPropertyTile('TPSA',
                          '${predictionResults['properties']['tpsa'].toStringAsFixed(2)} Å²'),
                      _buildPropertyTile(
                          'H-Bond Donors',
                          predictionResults['properties']['h_donors']
                              .toString()),
                      _buildPropertyTile(
                          'H-Bond Acceptors',
                          predictionResults['properties']['h_acceptors']
                              .toString()),
                      _buildPropertyTile(
                          'Rotatable Bonds',
                          predictionResults['properties']['rotatable_bonds']
                              .toString()),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // Helper method to build property display tiles
  Widget _buildPropertyTile(String title, String value) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF172A45),
        borderRadius: BorderRadius.circular(12),
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: GoogleFonts.roboto(
              fontSize: 14,
              color: Colors.white70,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: GoogleFonts.roboto(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  Widget buildAnalysisTab() {
    if (!showResult) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.insights,
              size: 80,
              color: Colors.white.withOpacity(0.3),
            ),
            const SizedBox(height: 24),
            Text(
              'Predict a chemical to see detailed analysis',
              style: GoogleFonts.roboto(
                fontSize: 18,
                color: Colors.white70,
              ),
            ),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Advanced Analysis',
            style: GoogleFonts.orbitron(
              fontSize: 28,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF64FFDA),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Detailed molecular properties for ${predictionResults['name']}',
            style: GoogleFonts.roboto(
              fontSize: 18,
              color: Colors.white70,
            ),
          ),
          const SizedBox(height: 32),
          Card(
            elevation: 12,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                children: [
                  _buildAnalysisRow('Molecular Weight',
                      '${predictionResults['properties']['molecular_weight'].toStringAsFixed(3)} g/mol'),
                  _buildDivider(),
                  _buildAnalysisRow(
                      'LogP (Partition Coefficient)',
                      predictionResults['properties']['mol_logp']
                          .toStringAsFixed(5)),
                  _buildDivider(),
                  _buildAnalysisRow('Topological Polar Surface Area',
                      '${predictionResults['properties']['tpsa'].toStringAsFixed(2)} Å²'),
                  _buildDivider(),
                  _buildAnalysisRow('Hydrogen Bond Donors',
                      predictionResults['properties']['h_donors'].toString()),
                  _buildDivider(),
                  _buildAnalysisRow(
                      'Hydrogen Bond Acceptors',
                      predictionResults['properties']['h_acceptors']
                          .toString()),
                  _buildDivider(),
                  _buildAnalysisRow('Ring Count',
                      predictionResults['properties']['ring_count'].toString()),
                  _buildDivider(),
                  _buildAnalysisRow(
                      'Rotatable Bonds',
                      predictionResults['properties']['rotatable_bonds']
                          .toString()),
                  _buildDivider(),
                  _buildAnalysisRow(
                      'Fraction Csp3',
                      predictionResults['properties']['fraction_csp3']
                          .toStringAsFixed(3)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 32),
          Card(
            elevation: 12,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Drug-likeness Analysis',
                    style: GoogleFonts.roboto(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 16),
                  LinearProgressIndicator(
                    value: 0.72,
                    backgroundColor: const Color(0xFF172A45),
                    valueColor:
                        const AlwaysStoppedAnimation<Color>(Color(0xFF64FFDA)),
                    minHeight: 12,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Lipinski Rule Compliance',
                        style: GoogleFonts.roboto(
                          fontSize: 14,
                          color: Colors.white70,
                        ),
                      ),
                      Text(
                        '3/4 Rules',
                        style: GoogleFonts.roboto(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: const Color(0xFF64FFDA),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  Row(
                    children: [
                      _buildRuleIndicator(
                          'MW ≤ 500',
                          predictionResults['properties']['molecular_weight'] <=
                              500),
                      const SizedBox(width: 16),
                      _buildRuleIndicator('LogP ≤ 5',
                          predictionResults['properties']['mol_logp'] <= 5),
                      const SizedBox(width: 16),
                      _buildRuleIndicator('HBD ≤ 5',
                          predictionResults['properties']['h_donors'] <= 5),
                      const SizedBox(width: 16),
                      _buildRuleIndicator('HBA ≤ 10',
                          predictionResults['properties']['h_acceptors'] <= 10),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAnalysisRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: Row(
        children: [
          Expanded(
            flex: 2,
            child: Text(
              label,
              style: GoogleFonts.roboto(
                fontSize: 16,
                fontWeight: FontWeight.w500,
                color: Colors.white,
              ),
            ),
          ),
          Expanded(
            flex: 1,
            child: Text(
              value,
              textAlign: TextAlign.end,
              style: GoogleFonts.robotoMono(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: const Color(0xFF64FFDA),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDivider() {
    return Divider(
      height: 1,
      color: Colors.white.withOpacity(0.1),
    );
  }

  Widget _buildRuleIndicator(String rule, bool passed) {
    return Column(
      children: [
        Container(
          width: 24,
          height: 24,
          decoration: BoxDecoration(
            color: passed ? const Color(0xFF64FFDA) : Colors.red,
            shape: BoxShape.circle,
          ),
          child: Center(
            child: Icon(
              passed ? Icons.check : Icons.close,
              size: 16,
              color: const Color(0xFF0A192F),
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          rule,
          style: GoogleFonts.roboto(
            fontSize: 12,
            color: Colors.white70,
          ),
        ),
      ],
    );
  }

  Widget buildTeamTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Our Team',
            style: GoogleFonts.orbitron(
              fontSize: 28,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF64FFDA),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'The brilliant minds behind ChemPredict Pro',
            style: GoogleFonts.roboto(
              fontSize: 18,
              color: Colors.white70,
            ),
          ),
          const SizedBox(height: 32),
          GridView.count(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisCount: 2,
            crossAxisSpacing: 24,
            mainAxisSpacing: 24,
            childAspectRatio: 0.8,
            children: [
              _buildTeamMember(
                name: 'Ajay Krishna',
                role: 'ML Engineer',
                image: 'assets/team1.jpg',
                linkedin: '#',
                github: '#',
              ),
              _buildTeamMember(
                name: 'Dashrad Raghav',
                role: 'ML Engineer',
                image: 'assets/team2.jpg',
                linkedin: '#',
                github: '#',
              ),
              _buildTeamMember(
                name: 'Bargavan',
                role: 'ML Engineer',
                image: 'assets/team3.jpg',
                linkedin: '#',
                github: '#',
              ),
              _buildTeamMember(
                name: 'Arun',
                role: 'Backend Engineer',
                image: 'assets/team4.jpg',
                linkedin: '#',
                github: '#',
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildTeamMember({
    required String name,
    required String role,
    required String image,
    required String linkedin,
    required String github,
  }) {
    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: const Color(0xFF64FFDA),
                  width: 2,
                ),
                color: const Color(0xFF172A45),
              ),
              child: Center(
                child: Icon(
                  Icons.person,
                  size: 60,
                  color: Color(0xFF64FFDA),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              name,
              style: GoogleFonts.roboto(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              role,
              style: GoogleFonts.roboto(
                fontSize: 14,
                color: const Color(0xFF64FFDA),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  icon: const Icon(Icons.link),
                  color: Colors.white70,
                  onPressed: () {},
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: const Icon(Icons.code),
                  color: Colors.white70,
                  onPressed: () {},
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
