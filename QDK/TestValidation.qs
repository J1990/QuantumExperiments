//Code taken from: https://github.com/microsoft/QuantumLibraries/blob/master/MachineLearning/src/Validation.qs
namespace TCD.MS.IS.Dissertation {
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.MachineLearning;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Logical;

    function Misclassifications(inferredLabels : Int[], actualLabels : Int[])
    : Int[] {
        return Where(
            NotEqualI,
            Zip(inferredLabels, actualLabels)
        );
    }

    function NMisclassifications(proposed: Int[], actual: Int[]): Int {
        return Length(Misclassifications(proposed, actual));
    }

    operation ValidateMnistModel1(
        validationVectors : Double[][],
        validationLabels : Int[],
        parameters : Double[],
        bias : Double
    ) : Int[] {
        // Get the remaining samples to use as validation data.
        let samples = Mapped(
            LabeledSample,
            Zip(Preprocessed(validationVectors), validationLabels)
        );
        let tolerance = 0.005;
        let nMeasurements = 10000;
        //let results = ValidateSequentialClassifier(
        //    SequentialModel(ClassifierStructure(), parameters, bias),
        //    samples,
        //    tolerance,
        //    nMeasurements,
        //    DefaultSchedule(samples)
        //);
        let model = SequentialModel(ClassifierStructure(), parameters, bias);
        let validationSchedule = DefaultSchedule(samples);
        let features = Mapped(_Features, samples);
        let labels = Sampled(validationSchedule, Mapped(_Label, samples));
        let probabilities = EstimateClassificationProbabilities(
            tolerance, model,
            Sampled(validationSchedule, features), nMeasurements
        );
        let localPL = InferredLabels(model::Bias, probabilities);
        let nMisclassifications = NMisclassifications(localPL, labels);
        let results = ValidationResults(
            nMisclassifications,
            Length(localPL)
        );

        //return IntAsDouble(results::NMisclassifications) / IntAsDouble(Length(samples));
        return localPL;
    }

    
    //operation ApplyHadamard(number : Int) : Int[]{
    //    let length = number;
    //    mutable result = new Int[length];
    //
    //    for (index in 0 .. length - 1) {
    //        using (qubit = Qubit()) {
    //            
    //            H(qubit);
    //            let measured = M(qubit);
    //            
    //            if(measured == One){
    //                set result w/= index <- 1;
    //            }
    //            else{
    //                set result w/= index <- 0;
    //            }
    //
    //            Reset(qubit);
    //        }
    //    }
        
    //    return result;
    //}
}