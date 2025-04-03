import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

class MLModel {
  late Map<String, dynamic> modelData;
  late Map<String, dynamic> scalerData;

  Future<void> loadModel() async {
    String modelJson = await rootBundle.loadString('assets/model.json');
    String scalerJson = await rootBundle.loadString('assets/scaler.json');

    modelData = jsonDecode(modelJson);
    scalerData = jsonDecode(scalerJson);
  }

  String predictAndFormat(String smiles, String compoundName) {
    if (modelData.isEmpty || scalerData.isEmpty) {
      return jsonEncode({"error": "Model not loaded!"});
    }

    // Simulated Prediction Logic (Replace this with actual computation)
    return jsonEncode({
      "Name": compoundName,
      "SMILES": smiles,
      "Predicted Properties": {
        "MolWt": modelData["MolWt"],
        "MolLogP": modelData["MolLogP"],
        "TPSA": modelData["TPSA"],
        "HDonors": modelData["HDonors"],
        "HAcceptors": modelData["HAcceptors"],
        "RingCount": modelData["RingCount"],
        "Rotatable Bonds": modelData["Rotatable Bonds"],
        "FractionCSP3": modelData["FractionCSP3"]
      },
      "Toxicity Prediction": modelData["Toxicity"],
      "Toxicity Confidence": modelData["ToxicityConfidence"]
    });
  }
}
