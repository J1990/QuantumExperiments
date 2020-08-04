using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;

namespace TCD.MS.IS.Dissertation
{
    class ClassicalHost
    {
        static async Task Main(string[] args)
        {
            try
            {
                string json_data_path = "Data\\MNIST\\mnist_pca_30Components_3_6.json";
                string logPath = "DebugOutput\\log.txt";

                Stopwatch stopwatch = new Stopwatch();

                stopwatch.Start();

                using(StreamWriter writer = File.AppendText(logPath))
                {
                    LogWriter.WriteToLog(writer, "Time: " + DateTime.Now);
                    LogWriter.WriteToLog(writer, "Time: " + DateTime.Now.TimeOfDay);

                    // We start by loading the training and validation data from our JSON
                    // data file.
                    var data = await LoadData(json_data_path);

                    // Next, we initialize a full state-vector simulator as our target machine.
                    using var targetMachine = new QuantumSimulator().WithTimestamps(writer);

                    // Once we initialized our target machine,
                    // we can then use that target machine to train a QCC classifier.
                    var (optimizedParameters, optimizedBias) = await TrainMnistModel.Run(
                        targetMachine,
                        new QArray<QArray<double>>(data.TrainingData.Features.Select(vector => new QArray<double>(vector))),
                        new QArray<long>(data.TrainingData.Labels)
                    );

                    // After training, we can use the validation data to test the accuracy
                    // of our new classifier.
                    var missRate = await ValidateMnistModel.Run(
                        targetMachine,
                        new QArray<QArray<double>>(data.ValidationData.Features.Select(vector => new QArray<double>(vector))),
                        new QArray<long>(data.ValidationData.Labels),
                        optimizedParameters,
                        optimizedBias
                    );
                    LogWriter.WriteToLog(writer, $"Observed {100 * missRate:F2}% misclassifications.");

                    stopwatch.Stop();
                    LogWriter.WriteToLog(writer, "Elapsed Total Minutes: " + stopwatch.Elapsed.TotalMinutes);

                }

                
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }


        }

        class LabeledData
        {
            public List<double[]> Features { get; set; }
            public List<long> Labels { get; set; }
        }

        class DataSet
        {
            public LabeledData TrainingData { get; set; }
            public LabeledData ValidationData { get; set; }
        }

        static async Task<DataSet> LoadData(string dataPath)
        {
            using var dataReader = File.OpenRead(dataPath);
            return await JsonSerializer.DeserializeAsync<DataSet>(
                dataReader
            );
        }
    }

    public static class SimulatorExtensions
    {
        public static QuantumSimulator WithTimestamps(this QuantumSimulator sim, TextWriter textWriter)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            var last = stopwatch.Elapsed;
            sim.DisableLogToConsole();
            sim.OnLog += (message) =>
            {
                var now = stopwatch.Elapsed;
                LogWriter.WriteToLog(textWriter, $"[{now} +{now - last}] {message}");
                last = now;
            };
            return sim;
        }
    }

    static class LogWriter
    {
        public static void WriteToLog(TextWriter writer, string message)
        {
            writer.WriteLine(message);
            Console.WriteLine(message);
        }
    }
}
