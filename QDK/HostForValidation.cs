using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;
using CsvHelper;
using System.Globalization;

namespace TCD.MS.IS.Dissertation.Validation
{
    class HostForValidation
    {
        static async Task Main(string[] args)
        {
            try
            {
                string json_data_path = "Data\\MNIST\\mnist_pca_10Components_3_6.json";

                Stopwatch stopwatch = new Stopwatch();

                stopwatch.Start();

                // We start by loading the training and validation data from our JSON
                // data file.
                var data = await LoadData(json_data_path);

                // Next, we initialize a full state-vector simulator as our target machine.
                using var targetMachine = new QuantumSimulator();

                double[] parameters = new double[] {-0.1973136176978195,-0.0592553467881118,-0.41513888957758055,-0.4458185053352934,-2.8409324191642074,-0.0337475773334841,-2.291767297101148,-1.1075341288534328,-0.7946020481247621,-0.5522330836388308,-2.0800779483729293,-2.9636508821950587};
                var optimizedParameters = new QArray<double>(parameters);
                double optimizedBias = -0.1158093451614467;



                var predictedLabels = (await ValidateMnistModel1.Run(
                    targetMachine,
                    new QArray<QArray<double>>(data.ValidationData.Features.Select(vector => new QArray<double>(vector))),
                    new QArray<long>(data.ValidationData.Labels),
                    optimizedParameters,
                    optimizedBias
                )).ToList();

                var pc1s = data.ValidationData.Features.Select(x=>x[0]).ToList();
                var pc2s = data.ValidationData.Features.Select(x=>x[1]).ToList();
                var pc3s = data.ValidationData.Features.Select(x=>x[2]).ToList();

                List<ResultRecord> records = new List<ResultRecord>();

                for(int i=0; i < pc1s.Count; i++){
                    records.Add(new ResultRecord(pc1s[i], pc2s[i], pc3s[i], data.ValidationData.Labels[i], predictedLabels[i]));
                }

                var path = "DebugOutput\\result.csv";

                using (StreamWriter writer = File.AppendText(path))
                {
                    var csvWriter = new CsvWriter(writer, CultureInfo.InvariantCulture);
                    csvWriter.WriteRecords(records);
                }

                stopwatch.Stop();
                Console.WriteLine("Elapsed Total Minutes: " + stopwatch.Elapsed.TotalMinutes);

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

        class ResultRecord
        {
            public double PC1 { get; set; }
            public double PC2 { get; set; }
            public double PC3 { get; set; }

            public long ActualLabel { get; set; }

            public long PredictedLabel { get; set; }

            public ResultRecord(double val1, double val2, double val3, long acLabel, long predLabel){
                PC1 = val1;
                PC2 = val2;
                PC3 = val3;
                ActualLabel = acLabel;
                PredictedLabel = predLabel;
            }
        }

        static async Task<DataSet> LoadData(string dataPath)
        {
            using var dataReader = File.OpenRead(dataPath);
            return await JsonSerializer.DeserializeAsync<DataSet>(
                dataReader
            );
        }
    }
}