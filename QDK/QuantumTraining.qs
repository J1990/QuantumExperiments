// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

namespace TCD.MS.IS.Dissertation {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.MachineLearning;
    open Microsoft.Quantum.Math;

    function WithProductKernel(scale : Double, sample : Double[]) : Double[] {
        return sample + [scale * Fold(TimesD, 1.0, sample)];
    }

    function Preprocessed(samples : Double[][]) : Double[][] {
        let scale = 1.0;

        return Mapped(
            WithProductKernel(scale, _),
            samples
        );
    }

    function DefaultSchedule(samples : LabeledSample[]) : SamplingSchedule {
        return SamplingSchedule([
            0..Length(samples) - 1
        ]);
    }

    function ClassifierStructure() : ControlledRotation[] {
        return CombinedStructure([
            LocalRotationsLayer(5, PauliZ),
            LocalRotationsLayer(5, PauliX),
            CyclicEntanglingLayer(5, PauliZ, 2)
        ]);
    }

    operation SampleSingleParameter() : Double {
        return PI() * (RandomReal(30) - 1.0);
    }

    operation SampleParametersForSequence(structure : ControlledRotation[]) : Double[] {
        return ForEach(SampleSingleParameter, ConstantArray(Length(structure), ()));
    }

    operation SampleInitialParameters(nInitialParameterSets : Int, structure : ControlledRotation[]) : Double[][] {
        return ForEach(SampleParametersForSequence, ConstantArray(nInitialParameterSets, structure));
    }

    operation TrainMnistModel(trainingVectors : Double[][],
        trainingLabels : Int[]) : (Double[], Double) {
        let samples = Mapped(
            LabeledSample,
            Zip(Preprocessed(trainingVectors), trainingLabels)
        );
        Message("Ready to train.");
        let structure = ClassifierStructure();
        // Sample a random set of parameters.
        let initialParameters = SampleInitialParameters(30, structure);

        let mappedModels = Mapped(
                SequentialModel(structure, _, 0.0),
                initialParameters
            );
        
        let (optimizedModel, nMisses) = TrainSequentialClassifier(
            mappedModels,
            samples,
            DefaultTrainingOptions()
                w/ LearningRate <- 0.4
                w/ MinibatchSize <- 100
                w/ Tolerance <- 0.01
                w/ NMeasurements <- 10000
                w/ MaxEpochs <- 2
                w/ VerboseMessage <- Message,
            DefaultSchedule(samples),
            DefaultSchedule(samples)
        );
        Message($"Training complete, found optimal parameters: {optimizedModel::Parameters}, Bias: {optimizedModel::Bias}");
        return (optimizedModel::Parameters, optimizedModel::Bias);
    }

}
