import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const DiagnosisApp = () => {
  const [model, setModel] = useState(null);
  const [symptoms, setSymptoms] = useState('');
  const [diagnosis, setDiagnosis] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Your dataset
  const dataset = [
    { disease: 'Fungal infection', symptoms: ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'] },
    { disease: 'Allergy', symptoms: ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'] },
    // Add more entries for other diseases
  ];

  useEffect(() => {
    const loadModel = async () => {
      try {
        setLoading(true);
        const loadedModel = await tf.loadLayersModel('../../../server/trained_model/model.json');//Have to run node trained
        setModel(loadedModel);
      } catch (error) {
        console.error('Error loading model:', error);
        setError('Error loading model. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    loadModel();

    // Clean up function to dispose the model on unmount
    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, []);

  const predictDiagnosis = async () => {
    if (!model) {
      console.error('Model not loaded!');
      return;
    }

    setLoading(true);

    // Convert user input symptoms to lower case
    const userSymptoms = symptoms.toLowerCase().split(',');

    // Find the disease with the highest probability based on the symptoms entered by the user
    let maxProbability = -1;
    let predictedDisease = '';

    dataset.forEach((entry) => {
      const entrySymptoms = entry.symptoms;
      const intersection = entrySymptoms.filter(symptom => userSymptoms.includes(symptom));
      const probability = intersection.length / entrySymptoms.length;

      if (probability > maxProbability) {
        maxProbability = probability;
        predictedDisease = entry.disease;
      }
    });

    setDiagnosis(predictedDisease);
    setLoading(false);
  };

  return (
    <div>
      <h1>Symptom Diagnosis</h1>
      <input type="text" value={symptoms} onChange={(e) => setSymptoms(e.target.value)} placeholder="Enter symptoms separated by commas" />
      <button onClick={predictDiagnosis} disabled={loading}>Diagnose</button>
      {loading && <p>Loading...</p>}
      {error && <p>Error: {error}</p>}
      {diagnosis && (
        <div>
          <h2>Diagnosis</h2>
          <p>{diagnosis}</p>
        </div>
      )}
    </div>
  );
};

export default DiagnosisApp;