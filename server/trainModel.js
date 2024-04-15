const tf = require('@tensorflow/tfjs-node');
const csv = require('csvtojson');

// Load the dataset from CSV
async function loadDataset() {
    const data = await csv().fromFile('dataset.csv');
    const symptoms = data.map(row => Object.values(row).slice(1, -1)); // Extract symptoms (excluding Disease column)
    const diagnoses = data.map(row => row.Disease); // Extract diagnoses
    return { symptoms, diagnoses };
}

// Encode categorical labels
function encodeLabels(labels) {
    const labelSet = new Set(labels);
    return labels.map(label => [...labelSet].indexOf(label));
}

async function trainModel() {
    const { symptoms, diagnoses } = await loadDataset();

    const X = tf.tensor(symptoms);
    const y = tf.tensor(encodeLabels(diagnoses));

    // Define and compile the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [X.shape[1]] }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    // Train the model
    await model.fit(X, y, {
        epochs: 10,
        batchSize: 32,
        validationSplit: 0.2,
    });

    // Save the trained model
    await model.save('file://trained_model');// This will generate trained_model/model.json
    console.log('Model saved successfully.');
}

// Train the model
trainModel().catch(console.error);