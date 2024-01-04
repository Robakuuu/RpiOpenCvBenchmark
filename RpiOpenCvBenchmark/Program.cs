using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace OpenCVBenchmarkRaspTest
{
    public class FaceDetection
    {
        private Mat _img;
        private CascadeClassifier _faceDetector;

        [GlobalSetup]
        public void Setup()
        {
            _img = CvInvoke.Imread("lena.jpg");
            _faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");
        }

        [Benchmark]
        public void DetectFace()
        {

            var imgGray = new UMat();
            var output = new UMat();
            CvInvoke.CvtColor(_img, imgGray, ColorConversion.Bgr2Gray);

            foreach (var face in _faceDetector.DetectMultiScale(imgGray, 1.1, 10, new Size(20, 20), Size.Empty))
            {
                CvInvoke.Rectangle(output, face, new MCvScalar(255, 255, 255));
            }
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
           var summary = BenchmarkRunner.Run<FaceDetection>();
        }
    }
}