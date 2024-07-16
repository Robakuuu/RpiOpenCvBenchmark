using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Microsoft.Diagnostics.Runtime.Utilities;
using Emgu.CV.Util;
using Microsoft.Diagnostics.Tracing.StackSources;
using UMapx.Imaging;
using Emgu.CV.Reg;

namespace OpenCVBenchmarkRaspTest
{
    public class OpenCVBenchmark
    {
        private Mat _img; 
        private Mat _firstFrame;
        private Mat _secondFrame;
        private Mat _dstHdr;
        private CascadeClassifier _faceDetector;
        private VectorOfMat _frames;

        private MergeMertens _merge_mertens;
        private ExposureFusion _fusion;
        private Bitmap[] _framesBitmap;
        [GlobalSetup]
        public void Setup()
        {
            _img = CvInvoke.Imread("lena.jpg");
            _faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");

            _firstFrame = CvInvoke.Imread("tig1.png");
            _secondFrame = CvInvoke.Imread("tig2.png");
         
            _frames = new VectorOfMat();
            _frames.Push(_firstFrame);
            _frames.Push(_secondFrame);
            _dstHdr = new Mat();
            _merge_mertens = new MergeMertens(1,0,0);
      
        }

     
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
       
     
        [Benchmark]
        public void SimpleHDR()
        {
            int width = _firstFrame.Width;
            int height = _secondFrame.Height;

            Image<Gray, Byte> firstImage = _firstFrame.ToImage<Gray, Byte>();
            Image<Gray, Byte> secondImage = _secondFrame.ToImage<Gray, Byte>();
            Image<Gray, Byte> dst = new Image<Gray, Byte>(width,height);
            for (int col = 0; col < width; col++)
            {
                for (int row = 0; row < height; row++)
                {
                    var avg = (firstImage.Data[row, col, 0] + secondImage.Data[row, col, 0]) / 2.0;
                    dst.Data[row, col, 0] = (byte)avg;
                }
            }
           // dst.Save("simpleHDR.png");
        }
        public void HdrTest()
        {
        
            _merge_mertens.Process(_frames, _dstHdr);
            _dstHdr = _dstHdr*255;
            CvInvoke.Imwrite("OpenCVHdr.png", _dstHdr);
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
       

             var summary = BenchmarkRunner.Run<OpenCVBenchmark>();
        }
    }
}