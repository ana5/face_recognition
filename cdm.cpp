#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include "opencv2/opencv.hpp"
//#include <boost/filesystem.hpp>
#include <dirent.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include "/home/aaina/Downloads/opencv-master/opencv_contrib/modules/face/include/opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include<stdlib.h>
#include <cstring>
 #include <unistd.h>
#include <string>
#include <sstream>
#include <opencv/cv.h>
#include <Python.h>
#include <ncurses.h>
using namespace cv;
using namespace cv::face;
using namespace std;
bool rotDefine = true;
bool createDir = false;

Mat imgr(Mat in,double angle);


Mat imgr(Mat im,double angle)
{
 Mat ds;
 Point2f pt(im.cols/2.,im.rows/2.);
 Mat r=getRotationMatrix2D(pt,angle,1.0);
 warpAffine(im,ds,r,Size(im.cols,im.rows));
cout<<"\t\t\t\t\tDelhi\n";
	return ds;
}



Mat cropFace(Mat srcImg, int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, int width, int height, int faceX, int faceY, int faceWidth, int faceHeight)
{
	Mat dstImg;
//	Mat crop;
	
if (!(eyeLeftX == 0 && eyeLeftY == 	0))
{
   int eyeXDirection = eyeRightX - eyeLeftX;
   int eyeYDirection = eyeRightY - eyeLeftY;
double angle = atan2((double)eyeYDirection, (double)eyeXDirection) * 180 /3.14159265;
	cerr<<"\t\t\tAngle\t\t"<<angle<<"\n";
   
     // dstImg = imgr(srcImg,angle);
	Point2f pt(srcImg.cols/2,srcImg.rows/2);
	Mat r=getRotationMatrix2D(pt,angle,1.0);
	 warpAffine(srcImg,dstImg,r,Size(srcImg.cols,srcImg.rows));
	
		
}
	//if(!dstImg.empty())
	return dstImg;
	
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') 
{
 std::ifstream file(filename.c_str(), ifstream::in);
 if (!file) 
 {
  string error_message = "No valid input file was given, please check the given filename.";
  CV_Error(CV_StsBadArg, error_message);
 }
 string line, path, classlabel;
 while (getline(file, line)) 
 {
  stringstream liness(line);
  getline(liness, path, separator);
  getline(liness, classlabel);
  if(!path.empty() && !classlabel.empty()) 
  {
   images.push_back(imread(path, 0));
   labels.push_back(atoi(classlabel.c_str()));
  }
 }
}
void trainface(string name)
{
	 string face_haar_cascade = string("/home/aaina/Downloads/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml");
 string eye_haar_cascade = string("/home/aaina/Downloads/opencv-3.4.1/data/haarcascades/haarcascade_eye.xml");
string csvFile = "/home/aaina/Desktop/FaceRecognition/csv.txt";
 //vector<Mat> images;
 vector<int> labels;
 CascadeClassifier face_cascade;
 CascadeClassifier eye_cascade;
 face_cascade.load(face_haar_cascade);
 eye_cascade.load(eye_haar_cascade);
 Mat frames;
 Mat original;
 Mat grey;
 vector<Mat> FaceData;
 vector<Rect > eyes;
 vector<Rect_<int> > faces;
 vector<int> eyesX;
 vector<int> eyesY;

 double fontScale = 2;
 int thickness = 1;
 int shift=0;
 /*try {
      read_csv(csvFile, images, labels);
     }catch (cv::Exception& e){
        cerr << "Error opening file \"" << csvFile << "\". Reason: " << e.msg << endl;
        exit(1);} */
	//int lab = labels[0];
	//cout<<"label\t"<<lab<<endl;
 int im_width = 100;
	//cout<<im_width<<endl;
 int im_height = 100;
	//cout<<im_width<<endl;
 	//Ptr<cv::face::FaceRecognizer> rocgniser = cv::face::LBPHFaceRecognizer::create();
 //rocgniser->train(images, labels);
 VideoCapture cap(0);
	
	Rect face_i2;
unsigned long n=0;
	string filepath= "/home/aaina/Desktop/FaceRecognition/Database/"+name;
	createDir = boost::filesystem::create_directories(filepath);
 while(cv::waitKey(1) != 27)
    { 
		try {
			cap.read(frames);
	    //cap.read(frames);
			if(!frames.empty())
			{	
		
      original=frames.clone();
      cvtColor(original,grey,CV_BGR2GRAY);
	 Rect face_i;
	 
   face_cascade.detectMultiScale(grey, faces, 1.1, 4,CV_HAAR_DO_CANNY_PRUNING, Size(frames.size().width*0.2, frames.size().height*0.2));
	//face_cascade.detectMultiScale(grey, faces,1.1,4,CV_HAAR_DO_CANNY_PRUNING);
			cout<<"Hello1\n";
			cout<<"_______"<<faces.size()<<"\n";
	    
	    for(size_t f=0;f<faces.size();f++)
        { cout<<"888888";
         face_i=faces[f];
			
		     int bottom_left_y = faces[f].y;
         if(bottom_left_y<0)
          { 
			     bottom_left_y=0;
		      }
         int top_right_y = faces[f].y + faces[f].height;
         if(top_right_y>frames.rows)
          {
			     top_right_y=frames.rows;
		      }  
         int bottom_left_x = faces[f].x;
         if(bottom_left_x<0)
          {
			     bottom_left_x=0;
		      }
         int top_right_x = faces[f].x + faces[f].width;
         if(top_right_x>frames.cols)
         {
			    top_right_x = frames.cols;
		     } 
			
         Point bottomLeftPoint(faces[f].x,bottom_left_y);
         Point topRightPoint(faces[f].x + faces[f].width,top_right_y);
         Rect faceArea(bottomLeftPoint,topRightPoint);
         Mat croppedFace_original= grey(faceArea);
         Mat croppedFace_original_rotated;
		// namedWindow("DEL",WINDOW_AUTOSIZE);
		// namedWindow("rotated",WINDOW_AUTOSIZE);
		//imshow("DEL",croppedFace_original);
         eye_cascade.detectMultiScale(croppedFace_original,eyes,1.1,3,CV_HAAR_DO_CANNY_PRUNING,Size(croppedFace_original.size().width*0.2,croppedFace_original.size().height*0.2));
         cout<<"Hello2\n";
			int eyeLeftX=0,eyeLeftY=0,eyeRightX=0,eyeRightY=0;
			
         for(size_t i=0;i<eyes.size();i++)
          { 
			  cout<<"<<<<<<<<<<<<<<<<<<<<<<Hello3>>>>>>>>>>>>>>>>>>>\t\t"<<i<<"\n";
			    int tleye2 = eyes[i].y + faces[f].y;
           if(tleye2<0)
            {
				tleye2=0;
			}
           int drEye2 = eyes[i].y + eyes[i].height + faces[f].y;
			  
           if(drEye2>frames.rows)
            {
				drEye2=frames.rows; 
			}
			  Point tl2(eyes[f].x + faces[i].x, tleye2);
			  Point dr2(eyes[f].x + eyes[f].width + faces[i].x, drEye2);
           if(eyeLeftX==0)
            {
             eyeLeftX=eyes[i].x;
			 eyeLeftY=eyes[i].y;
            }
          else if(eyeRightX==0)
            {
             eyeRightX= eyes[i].x;
			  eyeRightY= eyes[i].y;	
            }
			  cout<<"<<<<<<<<<<<<<<<"<<tl2<<"\n";
			  
		     
			  if(eyeLeftX>eyeRightX)
		   {
		   croppedFace_original_rotated = cropFace(croppedFace_original,eyeRightX,eyeRightY,eyeLeftX,eyeLeftY,200,200,faces[f].x,faces[f].y,faces[f].width,faces[f].height);
		   
		   }
		   else
		   {
		   croppedFace_original_rotated = cropFace(croppedFace_original,eyeLeftX,eyeLeftY,eyeRightX,eyeRightY,200,200,faces[f].x,faces[f].y,faces[f].width,faces[f].height);
		   } 
			 
		if(!croppedFace_original_rotated.empty())
		{ 
			//imshow("rotated",croppedFace_original_rotated);
		 
		
	        Mat face_resized;
		
            resize(croppedFace_original_rotated, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			 if(createDir)
		 { 
			 std::stringstream ss; 
             ss << n;
             std::string str = ss.str();
			string su= "/"+name+str;
			 string st=su+".jpg";
			string finalpath = filepath+st;
			
			 cout<<finalpath<<"\n";
		 bool saved =imwrite(finalpath,face_resized);
			 if(!saved)
				 cerr<<"\t\t\t\tUnable to save\n";
			 n+=1;
		 }
			
			//  int predictLabel = rocgniser->predict(face_resized);
			 
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            
            //string box_text = format("predictLabel = %d", predictLabel);
	      //int pos_x = std::max(face_i.tl().x - 10, 0);
            //int pos_y = std::max(face_i.tl().y - 10, 0);
         
            //putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
			//cout<<"Hello5\n";
		}
	 
		  } 
		  }
	 imshow("face_recognizer",original);
	 cout<<"Hello6\n";
	     //   waitKey(100);

   }
		}
		catch (Exception& e)
{
    const char* err_msg = e.what();
    std::cout << "exception caught: imshow:\n" << err_msg << std::endl;
}
		
	}
	system("xfce4-terminal -x sh -c 'cd ; cd /media/ashu/BigBang/Career/Opencv/Workspace/FaceRecognition/Database ; tree ; cd ..; python create_csv.py /media/ashu/BigBang/Career/Opencv/Workspace/FaceRecognition/Database; exec bash'");


}

void Recognise()
{
string face_haar_cascade = string("/media/ashu/BigBang/Career/Opencv/Workspace/haarcascade_frontalface_default.xml");
 string eye_haar_cascade = string("/media/ashu/BigBang/Career/Opencv/Workspace/haarcascade_eye.xml");
 string csvFile = "/media/ashu/BigBang/Career/Opencv/Workspace/FaceRecognition/csv.txt";
 vector<Mat> images;
 vector<int> labels;
 CascadeClassifier face_cascade;
 CascadeClassifier eye_cascade;
 face_cascade.load(face_haar_cascade);
 eye_cascade.load(eye_haar_cascade);
 Mat frames;
 Mat original;
 Mat grey;
 vector<Mat> FaceData;
 vector<Rect > eyes;
 vector<Rect_<int> > faces;
 vector<int> eyesX;
 vector<int> eyesY;

 double fontScale = 2;
 int thickness = 1;
 int shift=0;
 try {
      read_csv(csvFile, images, labels);
     }catch (cv::Exception& e){
        cerr << "Error opening file \"" << csvFile << "\". Reason: " << e.msg << endl;
        exit(1);}
 int im_width = images[0].cols;
 int im_height = images[0].rows;
 	Ptr<cv::face::FaceRecognizer> rocgniser = cv::face::LBPHFaceRecognizer::create(1,6,4,4,12.0);
 rocgniser->train(images, labels);
cout<<"hello\n";
 VideoCapture cap(0);

	Rect face_i2;
unsigned long n=0;
	//string filepath= "/home/ashu/Database";
	//createDir = boost::filesystem::create_directories(filepath);
	
 while(cv::waitKey(1) != 27)
    { ifstream nam;
	nam.open("Name.txt");
		try {
			cap.read(frames);
	    //cap.read(frames);
			if(!frames.empty())
			{	
		
      original=frames.clone();
      cvtColor(original,grey,CV_BGR2GRAY);
	 Rect face_i;
	 
   face_cascade.detectMultiScale(grey, faces, 1.1, 4,CV_HAAR_DO_CANNY_PRUNING, Size(frames.size().width*0.2, frames.size().height*0.2));
	//face_cascade.detectMultiScale(grey, faces,1.1,4,CV_HAAR_DO_CANNY_PRUNING);
			cout<<"Hello1\n";
			cout<<"_______"<<faces.size()<<"\n";
	    
	    for(size_t f=0;f<faces.size();f++)
        { cout<<"888888";
         face_i=faces[f];
			
		     int bottom_left_y = faces[f].y;
         if(bottom_left_y<0)
          { 
			     bottom_left_y=0;
		      }
         int top_right_y = faces[f].y + faces[f].height;
         if(top_right_y>frames.rows)
          {
			     top_right_y=frames.rows;
		      }  
         int bottom_left_x = faces[f].x;
         if(bottom_left_x<0)
          {
			     bottom_left_x=0;
		      }
         int top_right_x = faces[f].x + faces[f].width;
         if(top_right_x>frames.cols)
         {
			    top_right_x = frames.cols;
		     } 
			
         Point bottomLeftPoint(faces[f].x,bottom_left_y);
         Point topRightPoint(faces[f].x + faces[f].width,top_right_y);
         Rect faceArea(bottomLeftPoint,topRightPoint);
         Mat croppedFace_original= grey(faceArea);
         Mat croppedFace_original_rotated;
		// namedWindow("DEL",WINDOW_AUTOSIZE);
		 //namedWindow("rotated",WINDOW_AUTOSIZE);
		//imshow("DEL",croppedFace_original);
         eye_cascade.detectMultiScale(croppedFace_original,eyes,1.1,3,CV_HAAR_DO_CANNY_PRUNING,Size(croppedFace_original.size().width*0.2,croppedFace_original.size().height*0.2));
         cout<<"Hello2\n";
			int eyeLeftX=0,eyeLeftY=0,eyeRightX=0,eyeRightY=0;
			
         for(size_t i=0;i<eyes.size();i++)
          { 
			  cout<<"<<<<<<<<<<<<<<<<<<<<<<Hello3>>>>>>>>>>>>>>>>>>>\t\t"<<i<<"\n";
			    int tleye2 = eyes[i].y + faces[f].y;
           if(tleye2<0)
            {
				tleye2=0;
			}
           int drEye2 = eyes[i].y + eyes[i].height + faces[f].y;
			  
           if(drEye2>frames.rows)
            {
				drEye2=frames.rows; 
			}
			  Point tl2(eyes[f].x + faces[i].x, tleye2);
			  Point dr2(eyes[f].x + eyes[f].width + faces[i].x, drEye2);
           if(eyeLeftX==0)
            {
             eyeLeftX=eyes[i].x;
			 eyeLeftY=eyes[i].y;
            }
          else if(eyeRightX==0)
            {
             eyeRightX= eyes[i].x;
			  eyeRightY= eyes[i].y;	
            }
			  cout<<"<<<<<<<<<<<<<<<"<<tl2<<"\n";
			  
		     
			  if(eyeLeftX>eyeRightX)
		   {
		   croppedFace_original_rotated = cropFace(croppedFace_original,eyeRightX,eyeRightY,eyeLeftX,eyeLeftY,200,200,faces[f].x,faces[f].y,faces[f].width,faces[f].height);
		   
		   }
		   else
		   {
		   croppedFace_original_rotated = cropFace(croppedFace_original,eyeLeftX,eyeLeftY,eyeRightX,eyeRightY,200,200,faces[f].x,faces[f].y,faces[f].width,faces[f].height);
		   } 
			 
		if(!croppedFace_original_rotated.empty())
		{ 
			//imshow("rotated",croppedFace_original_rotated);
		 
		
	        Mat face_resized;
		
            resize(croppedFace_original_rotated, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			
			
			  int predictLabel = rocgniser->predict(face_resized);
			if(predictLabel==-1)
			{
			rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            
            string box_text = "unknown";
	      int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
         
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
			cout<<"Hello5\n";
			}
	else
	 {
	string line;
	stringstream ss;
      ss << predictLabel;
  string label = ss.str();
	string name;
	int point;
//	int count;
	while(getline(nam,line))
	{ 
		string lab;
	  point = 0;

	 cout<<"len\t"<<line.length()<<endl;
		for(int i=0; i<=line.length();i++)
	    {
			point++;
			if(line[i]==':')
			{	
			    cout<<"pos\t"<<point<<endl;
				break;}
			lab+=line[i];
			
			
		}
	// cout<<lab<<endl;
	 if(lab==label)
			{ //nam.seekp(point,ios::beg);
			
				for(int j=point;j<=line.length();j++)
					
				{ cout<<"Hello\n";
				 if(line[j]=='?')
					 break;
					name += line[j];
					
				}
				cout<<"\tNAME OF THE PERSON IS \t"<<name<<endl;
			
			}
	
	} 
	

			 
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            
            string box_text = name;
	      int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
         
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
			cout<<"Hello5\n";
	 } 
			
		}
	 
		  } 
		  }
	 imshow("face_recognizer",original);
				nam.close();
	 cout<<"Hello6\n";
	        //waitKey(100);
				

   }
		}
		catch (Exception& e)
{
    const char* err_msg = e.what();
    std::cout << "exception caught: imshow:\n" << err_msg << std::endl;
}
	}
	
	
}
string file()
 {
	
	string name;
	int label=0;
	int point;
	string line;
     ifstream nameFile;
 nameFile.open("Name.txt");
	 cout<<"Enter ur name \n";
	cin>>name;
	//cout<<"label \n";
	//cin>>label;
	while(getline(nameFile,line))
		{
			string lab;
	     point = 0;

	// cout<<"len\t"<<line.length()<<endl;
		for(int i=0; i<=line.length();i++)
	    {
			point++;
			if(line[i]==':')
			{	
			   // cout<<"pos\t"<<point<<endl;
				break;
			}
			lab+=line[i];
			
			
			
		}
			stringstream ss(lab);
			ss>>label;
			//cout<<label<<endl;
	    // cout<<lab<<endl;
			label = label+1;
			
		}
	
	nameFile.close();
	ofstream nam;
	nam.open("Name.txt",ios::app);
	
	std::stringstream ss; 
             ss << label;
             std::string str = ss.str();
	string st= str+":";
    nam<<st;
	//nameFile.seekp(2,ios::cur);
	string dp = name + "?";
	nam<<dp<<endl;
nam.close();
	
		 return name;
 }

int main()
{  int ch;
 string choice;
  String name; 
	//getch();
 HERE:
  	 																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					;
 cout<<"<<<<<<<<<<<<<<<<\t\t Say 'TRAIN ME' for training\t\t>>>>>>>>>>>>>>\n";
 cout<<"<<<<<<<<<<<<<<\t\tSay 'RECOGNISE ME' for  Recognition\t\t>>>>>>>>>>>>>>\n";
 cout<<"<<<<<<<<<<<<<<\t\tSay 'EXIT' for exit\t\t>>>>>>>>>>>>>>\n\n";
 //cout<<"<<<<<<<<<<<<<<\t\t Press 4 for new face Training\t>>>>>>>\n";
cout<<"TYPE PASSWORD TO ENTER \n";
 cin>>ch; 
// cin>>ch;
  // Set PYTHONPATH TO working directory
   setenv("PYTHONPATH",".",1);

   PyObject *pName, *pModule, *pDict, *pFunc, *pValue, *presult;


   // Initialize the Python Interpreter
   Py_Initialize();


   // Build the name object
   pName = PyString_FromString((char*)"speech");

   // Load the module object
   pModule = PyImport_Import(pName);


   // pDict is a borrowed reference 
   pDict = PyModule_GetDict(pModule);


   // pFunc is also a borrowed reference 
   pFunc = PyDict_GetItemString(pDict, (char*)"sp");

   if (PyCallable_Check(pFunc))
   {
       pValue=Py_BuildValue("(z)",(char*)"something");
       PyErr_Print();
      // printf("Let's give this a shot!\n");
	  cout<<"Let's give this a shot!\n";
       presult=PyObject_CallObject(pFunc,pValue);
       PyErr_Print();
   } else 
   {
       PyErr_Print();
   }
  // printf("Result is %zu\n",PyInt_AsLong(presult));
	//cout<<"C++ REsult\t"<<PyString_AsString(presult);
   choice= PyString_AsString(presult);
   Py_DECREF(pValue);

   // Clean up
   Py_DECREF(pModule);
   Py_DECREF(pName);

   // Finish the Python Interpreter
   Py_Finalize();
 if(choice == "train me")
 {
	            name= file();
		       trainface(name);
		       goto HERE;
 
 }
 else if (choice =="recognise me")
 {
              Recognise();
		      goto HERE;
 } 
 else if(choice == "exit")
 {
  exit(1);
 }
 else
 {
 cout<<"Wrong Choice\n"; 
 }
return 0;
}


