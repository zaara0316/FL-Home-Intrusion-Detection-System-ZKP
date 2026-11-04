
pragma circom 2.2.3;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/bitify.circom";
include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/gates.circom";

template FedNatDInference() {

    // ------------------------------------------------
    // PUBLIC INPUTS
    // ------------------------------------------------

    signal input modelCommitment;
    signal input prediction;

    // ------------------------------------------------
    // PRIVATE INPUTS
    // ------------------------------------------------

    // Quantized feature magnitudes and signs
    signal input featureMag[115];
    signal input featureSign[115];

    // Quantized model weight magnitudes and signs
    signal input weightMag[115];
    signal input weightSign[115];

    // Quantized bias
    signal input biasMag;
    signal input biasSign;

    // ------------------------------------------------
    // MODEL COMMITMENT
    // ------------------------------------------------

    // Signed field encoding:
    // Positive value -> magnitude
    // Negative value -> field modulus - magnitude

    signal encodedWeights[115];
    signal encodedBias;

    signal hashChain[116];

    var FIELD_MODULUS =
        21888242871839275222246405745257275088548364400416034343698204186575808495617;

    var DOMAIN = 20261104;

    for (var i = 0; i < 115; i++) {

        featureSign[i] * (featureSign[i] - 1) === 0;
        weightSign[i] * (weightSign[i] - 1) === 0;

        encodedWeights[i] <== weightMag[i] +
            weightSign[i] * (FIELD_MODULUS - 2 * weightMag[i]);

    }

    biasSign * (biasSign - 1) === 0;

    encodedBias <== biasMag +
        biasSign * (FIELD_MODULUS - 2 * biasMag);

    // First Poseidon hash includes a domain separator
    component firstHash = Poseidon(2);

    firstHash.inputs[0] <== DOMAIN;
    firstHash.inputs[1] <== encodedWeights[0];

    hashChain[0] <== firstHash.out;

    // Hash the remaining weights
    component weightHashes[114];

    for (var i = 1; i < 115; i++) {

        weightHashes[i-1] = Poseidon(2);

        weightHashes[i-1].inputs[0] <== hashChain[i-1];
        weightHashes[i-1].inputs[1] <== encodedWeights[i];

        hashChain[i] <== weightHashes[i-1].out;
    }

    // Include the bias in the commitment
    component biasHash = Poseidon(2);

    biasHash.inputs[0] <== hashChain[114];
    biasHash.inputs[1] <== encodedBias;

    hashChain[115] <== biasHash.out;

    // The calculated commitment must equal the public commitment
    hashChain[115] === modelCommitment;

    // ------------------------------------------------
    // RANGE CONSTRAINTS
    // ------------------------------------------------

    component featureBits[115];
    component weightBits[115];

    for (var i = 0; i < 115; i++) {

        // Features: 19-bit magnitudes
        featureBits[i] = Num2Bits(19);
        featureBits[i].in <== featureMag[i];

        // Weights: 10-bit magnitudes
        weightBits[i] = Num2Bits(10);
        weightBits[i].in <== weightMag[i];
    }

    // Bias magnitude: 7 bits
    component biasBits = Num2Bits(7);
    biasBits.in <== biasMag;

    // ------------------------------------------------
    // FIXED-POINT INFERENCE
    // ------------------------------------------------

    // Scale = 10000
    // Feature * weight gives scale 100000000.
    // Bias is multiplied by 10000 to match that scale.

    signal products[115];
    signal productSigns[115];

    signal positiveTerms[115];
    signal negativeTerms[115];

    component productBits[115];

    component signXor[115];

    for (var i = 0; i < 115; i++) {

        products[i] <== featureMag[i] * weightMag[i];

        // Maximum product fits within 29 bits
        productBits[i] = Num2Bits(29);
        productBits[i].in <== products[i];

        // XOR determines whether product is negative
        signXor[i] = XOR();

        signXor[i].a <== featureSign[i];
        signXor[i].b <== weightSign[i];

        productSigns[i] <== signXor[i].out;

        positiveTerms[i] <== products[i] * (1 - productSigns[i]);

        negativeTerms[i] <== products[i] * productSigns[i];
    }

    // ------------------------------------------------
    // ACCUMULATE POSITIVE AND NEGATIVE CONTRIBUTIONS
    // ------------------------------------------------

    signal positiveAcc[116];
    signal negativeAcc[116];

    positiveAcc[0] <== 0;
    negativeAcc[0] <== 0;

    for (var i = 0; i < 115; i++) {

        positiveAcc[i+1] <== positiveAcc[i] + positiveTerms[i];

        negativeAcc[i+1] <== negativeAcc[i] + negativeTerms[i];
    }

    // Bias contribution:
    // biasMag * 10000

    signal biasContribution;

    biasContribution <== biasMag * 10000;

    signal positiveTotal;
    signal negativeTotal;

    positiveTotal <== positiveAcc[115] +
        biasContribution * (1 - biasSign);

    negativeTotal <== negativeAcc[115] +
        biasContribution * biasSign;

    // Both totals must fit within 36 bits
    component positiveTotalBits = Num2Bits(36);
    component negativeTotalBits = Num2Bits(36);

    positiveTotalBits.in <== positiveTotal;
    negativeTotalBits.in <== negativeTotal;

    // ------------------------------------------------
    // CLASSIFICATION
    // ------------------------------------------------

    // Prediction = 1 when positiveTotal >= negativeTotal
    // Prediction = 0 otherwise

    component comparison = LessThan(36);

    comparison.in[0] <== positiveTotal;
    comparison.in[1] <== negativeTotal;

    prediction === 1 - comparison.out;

    prediction * (prediction - 1) === 0;
}

component main {public [modelCommitment, prediction]} = FedNatDInference();
