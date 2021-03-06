﻿using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Text;

namespace TCD.MS.IS.Dissertation
{
    class ClassicalHost
    {
        static async Task Main(string[] args)
        {
            try
            {
                string json_data_path = "Data\\MNIST\\mnist_pca_10Components_3_6.json";

                Stopwatch stopwatch = new Stopwatch();

                stopwatch.Start();

                LogWriter.WriteToLog("Time: " + DateTime.Now);
                LogWriter.WriteToLog("Time: " + DateTime.Now.TimeOfDay);

                // We start by loading the training and validation data from our JSON
                // data file.
                var data = await LoadData(json_data_path);

                // Next, we initialize a full state-vector simulator as our target machine.
                using var targetMachine = new QuantumSimulator().WithTimestamps();

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
                LogWriter.WriteToLog($"Observed {100 * missRate:F2}% misclassifications.");

                stopwatch.Stop();
                LogWriter.WriteToLog("Elapsed Total Minutes: " + stopwatch.Elapsed.TotalMinutes, true);

            }
            catch (Exception ex)
            {
                LogWriter.WriteToLog("Error: " + ex.Message, true);
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
        public static QuantumSimulator WithTimestamps(this QuantumSimulator sim)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            var last = stopwatch.Elapsed;
            sim.DisableLogToConsole();
            sim.OnLog += (message) =>
            {
                var now = stopwatch.Elapsed;
                LogWriter.WriteToLog($"[{now} +{now - last}] {message}");
                last = now;
            };
            return sim;
        }
    }

    static class LogWriter
    {
        static int count = 0;
        static string logPath = "DebugOutput\\log.txt";

        static StringBuilder stringBuilder = new StringBuilder();

        public static void WriteToLog(string message, bool forceWrite = false)
        {
            stringBuilder.AppendLine(message);
            Console.WriteLine(message);

            count++;

            if (forceWrite || count > 999)
            {
                using (StreamWriter writer = File.AppendText(logPath))
                {
                    writer.WriteLine(stringBuilder.ToString());
                }

                stringBuilder.Clear();
                count = 0;
            }
        }
    }
}
