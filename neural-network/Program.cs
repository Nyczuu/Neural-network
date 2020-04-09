using System;
using System.Collections.Generic;
using System.Linq;

namespace neural_network
{
    class Program
    {
        private static MathService _service;
        private static int _epochs = 50;

        static void Main(string[] args)
        {
            _service = new MathService();

            var input = new double[]
            {
                5.1, 3.5, 1.4, 0.2,
                4.9, 3.0, 1.4, 0.2,
                6.2, 3.4, 5.4, 2.3,
                5.9, 3.0, 5.1, 1.8
            };

            var output = new double[]
            {
                0,
                0,
                1,
                1
            };

            var predictedOutput = new double[output.Length];

            var weights = new double[]
            {
                0.5,
                0.5,
                0.5,
                0.5
            };

            for (var i = 0; i < _epochs; i++)
            {
                var predictions = _service.Sigmoid(_service.Product(input, weights, 4, 4, 1));
                var predictionsError = _service.Subtract(output, predictions);
                var predictionsDelta = _service.Multiply(predictionsError, _service.SigmoidDerivative(predictions));
                var weightsDelta = _service.Product(_service.Transpose(input, 4, 4), predictionsDelta, 4, 4, 1);

                weights = _service.Add(weights, weightsDelta);
                predictedOutput = predictions;

            }

            output.Print("Expected results");
            predictedOutput.Print("Actual results");
        }
    }

    public static class ArrayExtensions
    {
        public static void Print(this double[] matrix, string message)
        {
            Console.WriteLine($"{message}:");
            foreach (var item in matrix)
                Console.WriteLine(item);
            Console.WriteLine(Environment.NewLine);
        }
    }

    public class MathService
    {
        public double[] SigmoidDerivative(double[] matrix) => matrix.Select(x => x * (1 - x)).ToArray();

        public double[] Sigmoid(double[] matrix) => matrix.Select(x => 1 / (1 + Math.Exp(-x))).ToArray();

        int GetGreaterLenght(double[] matrix1, double[] matrix2) => matrix1.Length > matrix2.Length ? matrix1.Length : matrix2.Length;
        int GetSmallerLenght(double[] matrix1, double[] matrix2) => matrix1.Length < matrix2.Length ? matrix1.Length : matrix2.Length;

        public double[] Add(double[] matrix1, double[] matrix2)
        {
            var max = GetGreaterLenght(matrix1, matrix2);
            var min = GetSmallerLenght(matrix1, matrix2);
            var result = new double[max];

            for (var i = 0; i < min; i++)
                result[i] = matrix1[i] + matrix2[i];

            return result;
        }

        public double[] Subtract(double[] matrix1, double[] matrix2)
        {
            var max = GetGreaterLenght(matrix1, matrix2);
            var min = GetSmallerLenght(matrix1, matrix2);
            var result = new double[max];

            for (var i = 0; i < min; i++)
                result[i] = matrix1[i] - matrix2[i];

            return result;
        }

        public double[] Multiply(double[] matrix1, double[] matrix2)
        {
            var max = GetGreaterLenght(matrix1, matrix2);
            var min = GetSmallerLenght(matrix1, matrix2);
            var result = new double[max];

            for (var i = 0; i < min; i++)
                result[i] = matrix1[i] * matrix2[i];

            return result;
        }

        public double[] Product(double[] matrix1, double[] matrix2, int matrix1Rows, int matrix1Columns, int matrix2Columns)
        {
            var result = new double[matrix1Rows * matrix2Columns];

            for (int row = 0; row < matrix1Rows; row++)
            {
                for (int col = 0; col < matrix2Columns; col++)
                {
                    for (int k = 0; k < matrix1Columns; k++)
                    {
                        result[row * matrix2Columns + col] += matrix1[row * matrix1Columns + k] * matrix2[k * matrix2Columns + col];
                    }
                }
            }


            return result;
        }

        public double[] Transpose(double[] matrix, int columns, int rows)
        {
            var max = columns * rows;
            var result = new double[max];

            for (int i = 0; i < max; i++)
            {
                var x = i / columns;
                var y = i % columns;
                result[i] = matrix[rows * y + x];
            }

            return result;
        }
    }
}
