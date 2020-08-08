
// OpencvHW2Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "OpencvHW2.h"
#include "OpencvHW2Dlg.h"
#include "afxdialogex.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "opencv2\opencv.hpp"
#include "stdio.h"
#include "stdlib.h"
#include "string"
#include "iostream"

using namespace cv;
using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 對 App About 使用 CAboutDlg 對話方塊

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// COpencvHW2Dlg 對話方塊



COpencvHW2Dlg::COpencvHW2Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_OPENCVHW2_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void COpencvHW2Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO1, EMComboBox);
	DDX_Control(pDX, IDC_COMBO1, EMComboBox);
}

BEGIN_MESSAGE_MAP(COpencvHW2Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &COpencvHW2Dlg::OnBnClickedOriginalHistogram)
	ON_BN_CLICKED(IDC_BUTTON2, &COpencvHW2Dlg::OnBnClickedEqualizedHistogram)
	ON_BN_CLICKED(IDC_BUTTON3, &COpencvHW2Dlg::OnBnClickedHoughCircle)
	ON_BN_CLICKED(IDC_BUTTON4, &COpencvHW2Dlg::OnBnClickedHueHistogram)
	ON_BN_CLICKED(IDC_BUTTON5, &COpencvHW2Dlg::OnBnClickedBackProjection)
	ON_BN_CLICKED(IDC_BUTTON6, &COpencvHW2Dlg::OnBnClickedCornerDetection)
	ON_BN_CLICKED(IDC_BUTTON7, &COpencvHW2Dlg::OnBnClickedIntrinsicMatrix)
	ON_BN_CLICKED(IDC_BUTTON8, &COpencvHW2Dlg::OnBnClickedExtrinsicMatrix)
	ON_BN_CLICKED(IDC_BUTTON9, &COpencvHW2Dlg::OnBnClickedDistortionMatrix)
	ON_BN_CLICKED(IDC_BUTTON10, &COpencvHW2Dlg::OnBnClickedAugmentedReality)
END_MESSAGE_MAP()


// COpencvHW2Dlg 訊息處理常式

BOOL COpencvHW2Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO: 在此加入額外的初始設定
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	CString numInComboBox;
	numInComboBox.Empty();
	for (int i = 0; i < 15; i++)
	{
		numInComboBox.Format(_T("Picture : %d"), i + 1);
		EMComboBox.AddString(numInComboBox);
	}

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void COpencvHW2Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。

void COpencvHW2Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。
HCURSOR COpencvHW2Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void COpencvHW2Dlg::OnBnClickedOriginalHistogram()//1-1畫出直方圖
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat OHinput = imread("images\\plant.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("OHinputphoto", OHinput);
	int histSize = 256;//直方圖bin數(橫軸)
	float range[] = { 0, 255 };//統計的範圍
	const float* histRange = { range };
	Mat histImg;
	Mat OHshowHistImg(256, 256, CV_8UC1, Scalar(255));  //存放統計數據的直方圖影像
	calcHist(&OHinput, 1, 0, Mat(), histImg, 1, &histSize, &histRange);//統計圖片資料
	float histMaxValue = 0;//存放統計灰階數據的最大值
	for (int i = 0; i<histSize; i++)//找出最大值
	{
		float tempValue = histImg.at<float>(i);
		if (histMaxValue < tempValue) {
			histMaxValue = tempValue;
		}
	}

	float scale = (0.9 * 256) / histMaxValue;//圖片縮放比例
	for (int i = 0; i<histSize; i++)//畫線
	{
		int intensity = static_cast<int>(histImg.at<float>(i)*scale);
		line(OHshowHistImg, Point(i, 255), Point(i, 255 - intensity), Scalar(0));
	}
	imshow("OHshowHist", OHshowHistImg);
	waitKey(0);
	destroyAllWindows();
}


void COpencvHW2Dlg::OnBnClickedEqualizedHistogram()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat EHinput = imread("images\\plant.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat EHoutput;
	equalizeHist(EHinput, EHoutput);//將圖片資訊作等化
	imshow("EHOutput", EHoutput);
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	Mat histImg;
	calcHist(&EHoutput, 1, 0, Mat(), histImg, 1, &histSize, &histRange);

	Mat EHshowHistImg(256, 256, CV_8UC1, Scalar(255));  //存放統計數據的直方圖影像
	float histMaxValue = 0;
	for (int i = 0; i<histSize; i++)
	{
		float tempValue = histImg.at<float>(i);
		if (histMaxValue < tempValue) {
			histMaxValue = tempValue;
		}
	}

	float scale = (0.9 * 256) / histMaxValue;
	for (int i = 0; i<histSize; i++)
	{
		int intensity = static_cast<int>(histImg.at<float>(i)*scale);
		line(EHshowHistImg, Point(i, 255), Point(i, 255 - intensity), Scalar(0));
	}
	imshow("EHshowHist", EHshowHistImg);
	waitKey(0);
	destroyAllWindows();
}


void COpencvHW2Dlg::OnBnClickedHoughCircle()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat HCInput = imread("images\\q2_train.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat HCOutput;
	HCOutput = HCInput.clone();
	vector<Vec3f> HCCircles;//儲存圓心半徑
	Mat HCsrcGray;
	cvtColor(HCInput, HCsrcGray, CV_BGR2GRAY);
	GaussianBlur(HCsrcGray, HCsrcGray, Size(9, 9), 2, 2);//先對圖像作銳化處理去除雜訊
	HoughCircles(HCsrcGray, HCCircles, CV_HOUGH_GRADIENT, 1.5, 30, 50, 32, 15, 20);//找圓
	for (int i = 0; i < HCCircles.size(); i++)//將圖像的圓形圈起來
	{
		Point center(cvRound(HCCircles[i][0]), cvRound(HCCircles[i][1]));
		int radius = cvRound(HCCircles[i][2]);
		//cout << cvRound(HCCircles[i][0]) << "\t" << cvRound(HCCircles[i][1]) << "\t" << cvRound(HCCircles[i][2]) << endl;
		circle(HCOutput, center, radius, Scalar(0, 255, 0), 2, 8, 0);//畫圓(圖，中心，半徑，顏色，邊界大小，8連通)
		HCOutput.at<Vec3b>(center)[0] = 0;//對圓心著紅色
		HCOutput.at<Vec3b>(center)[1] = 0;
		HCOutput.at<Vec3b>(center)[2] = 255;
	}
	imshow("InputImage", HCInput);
	imshow("OutputImage", HCOutput);
	waitKey(0);
	destroyAllWindows();
}


void COpencvHW2Dlg::OnBnClickedHueHistogram()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat HHInput = imread("images\\q2_train.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat HHsrcGray;
	cvtColor(HHInput, HHsrcGray, CV_BGR2GRAY);
	Mat HHCirclesImg(HHInput.rows, HHInput.cols, CV_8UC3, Scalar(0));
	Mat HHCirclesHSVImg;
	vector<Vec3f> HHCircles;//儲存找到的圓心及半徑
	GaussianBlur(HHsrcGray, HHsrcGray, Size(9, 9), 2, 2);
	//1.5, 15, 50, 24, 8, 12   test圖參數
	//1.5, 30, 50, 32, 15, 20 train圖參數
	HoughCircles(HHsrcGray, HHCircles, CV_HOUGH_GRADIENT, 1.5, 30, 50, 32, 15, 20);
	for (int i = 0; i < HHCircles.size(); i++)
	{
		Point center(cvRound(HHCircles[i][0]), cvRound(HHCircles[i][1]));
		int radius = cvRound(HHCircles[i][2]);
		circle(HHCirclesImg, center, radius, Scalar(HHInput.at<Vec3b>(center)[0], HHInput.at<Vec3b>(center)[1], HHInput.at<Vec3b>(center)[2]), CV_FILLED, 8, 0);
	    //將找到的圓存到一張新的圖像中，新的圖片只保存這些圓形
	}
	imshow("HHInput", HHInput);
	//imshow("HHCirclesTest",HHCirclesImg);

	cvtColor(HHCirclesImg, HHCirclesHSVImg, CV_BGR2HSV);
	int histSize = 256;
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };
	Mat histImg;//儲存統計結果
	Mat HHshowHistImg(256, 256, CV_8UC1, Scalar(255));  //將統計結果轉換成一張直方圖
	calcHist(&HHCirclesHSVImg, 1, channels, Mat(), histImg, 1, &histSize, ranges);
	float histMaxValue = 0;
	for (int i = 1; i<histSize; i++)//統計最大值
	{
		float tempValue = histImg.at<float>(i);
		if (histMaxValue < tempValue) {
			histMaxValue = tempValue;
		}
	}

	float scale = (0.9 * 256) / histMaxValue;
	for (int i = 1; i<histSize; i++)//畫圓
	{
		int intensity = static_cast<int>(histImg.at<float>(i)*scale);
		line(HHshowHistImg, Point(i, 255), Point(i, 255 - intensity), Scalar(0));
	}
	imshow("HHshowHist", HHshowHistImg);
	waitKey(0);
	destroyAllWindows();
}


void COpencvHW2Dlg::OnBnClickedBackProjection()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat BPtrainInput = imread("images\\q2_train.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat BPtestInput= imread("images\\q2_test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat BPtestHSV;
	cvtColor(BPtestInput, BPtestHSV, CV_BGR2HSV);
	Mat BPsrcGray;//將圖片train 轉成灰階圖用來做HoughCirclr
	cvtColor(BPtrainInput, BPsrcGray, CV_BGR2GRAY);
	Mat BPCirclesImg(BPtrainInput.rows, BPtrainInput.cols, CV_8UC3, Scalar(0));
	Mat BPCirclesHSVImg;
	vector<Vec3f> BPCircles;
	GaussianBlur(BPsrcGray, BPsrcGray, Size(9, 9), 2, 2);
	HoughCircles(BPsrcGray, BPCircles, CV_HOUGH_GRADIENT, 1.5, 30, 50, 32, 15, 20);
	for (int i = 0; i < BPCircles.size(); i++)//將找到的圓重新畫在一個新的圖片中
	{
		Point center(cvRound(BPCircles[i][0]), cvRound(BPCircles[i][1]));
		int radius = cvRound(BPCircles[i][2]);
		circle(BPCirclesImg, center, radius, Scalar(BPtrainInput.at<Vec3b>(center)[0], BPtrainInput.at<Vec3b>(center)[1], BPtrainInput.at<Vec3b>(center)[2]), CV_FILLED, 8, 0);
	}
	
	//將找到的圓形儲存圖片轉換成HSV圖
	cvtColor(BPCirclesImg, BPCirclesHSVImg, CV_BGR2HSV);
	//imshow("BPHSVTest", BPCirclesHSVImg);
	Mat BPHistImg;
	int hbins = 2, sbins = 2;
	int srcHistSize[] = { hbins, sbins };
	float ranges1[] = { 103, 121 };
	float ranges2[] = { 48, 190 };
	const float* BackProjectRange[] = { ranges1 ,ranges2 };
	int HSVChannel[] = { 0, 1 };
	Mat srcBackProject;	
	calcHist(&BPCirclesHSVImg, 1, HSVChannel, Mat(), BPHistImg, 2, srcHistSize, BackProjectRange);
	normalize(BPHistImg, BPHistImg, 0, 255, NORM_MINMAX);
	//把HSV圖作成Histogram並規一化
	calcBackProject(&BPtestHSV, 1, HSVChannel, BPHistImg, srcBackProject, BackProjectRange, 255, true);
	//normalize(srcBackProject, srcBackProject, 0, 255, NORM_MINMAX);
	//利用規一化的數據放入要偵測的圖片中，判斷圖片是否符合此直方圖，出來的數據為機率值位於0-1之間
	//如果將此機率數據規一化0-255，代表白色部分越符合此直方圖
	imshow("BPtestInput", BPtestInput);
	imshow("BackProjection_Result", srcBackProject);
	waitKey(0);
	destroyAllWindows();
}

Mat CCInput[15];//存放15張圖片
Size cornerSize = Size(11, 8);//棋盤格大小
vector<Point2f> CCcorners;//儲存找到的corner
bool CCDetectResult[15];//儲存corner偵測結果 值為0或1
bool CCFunctionLIFlag = 0;//控制載入圖片的旗標
bool CCFunctionCDFlag = 0;//控制角度偵測的旗標
bool CCFunctionCalFlag = 0;//控制校準的旗標

vector<vector<Point2f>> IMImagePoint;
vector<vector<Point3f>> IMObjectPoint;
Mat CCIntrinsic = Mat(3, 3, CV_32FC1);//內參
Mat CCDistCoeffs;//儲存因為透鏡及組裝技術誤差的校正參數
vector<Mat> rvecs;//儲存Rotation參數
vector<Mat> tvecs;//儲存translation參數
int EMSelect = 0;//儲存2-3所選擇的圖片

Mat ARInput[5];

void CCLoadingInput()//載入圖片
{
	int i = 0;
	String photoPath = "images\\CameraCalibration\\";
	//TermCriteria criteria[15];
	//char Name[64];
	cout << "Loading Images for First Time" << endl;

	for (i = 0; i < 15; i++)
	{
		//sprintf(Name, "%d", i+1);
		CCInput[i] = imread(photoPath + to_string(i + 1) + ".bmp", CV_LOAD_IMAGE_UNCHANGED);
		cout << photoPath + to_string(i + 1) + ".bmp" << endl;
		if (i < 5)
		{
			ARInput[i] = imread(photoPath + to_string(i + 1) + ".bmp", CV_LOAD_IMAGE_UNCHANGED);
		}
	}
	CCFunctionLIFlag = 1;
	cout << "Loading Completed\n" << endl;
}

void CCCornerDetection()//角度偵測
{
	int i = 0;
	cout << "Start 15 Images Corner Detection , please wait a few seconds" << endl;
	cout << "In the Corner Detection Status..." << endl;
	for (i = 0; i < 15; i++)
	{
		CCDetectResult[i] = findChessboardCorners(CCInput[i], cornerSize, CCcorners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
		IMImagePoint.push_back(CCcorners);
		//criteria[i] = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.1);
		//cornerSubPix(CCInput[i], CCcorners, Size(5, 5), Size(-1, -1), criteria[i]);
		drawChessboardCorners(CCInput[i], cornerSize, CCcorners, CCDetectResult[i]);
	}
	CCFunctionCDFlag = 1;
	cout << "Corner Detection Completed\n" << endl;
}

void CCCalibration()//校準
{
	int numCors = cornerSize.width*cornerSize.height;
	vector<Point3f> chessborad_pts;
	int i = 0;
	cout << "Start Calibration..." << endl;
	for (i = 0; i<numCors; i++)
	{
		chessborad_pts.push_back(Point3f(i / cornerSize.width, i%cornerSize.width, 0.0f));//建立3D點座標
	}
	for (i = 0; i<15; i++)
	{
		IMObjectPoint.push_back(chessborad_pts);
	}
	calibrateCamera(IMObjectPoint, IMImagePoint, cornerSize, CCIntrinsic, CCDistCoeffs, rvecs, tvecs);
	CCFunctionCalFlag = 1;
	cout << "Calibration Completed\n" << endl;
}

void COpencvHW2Dlg::OnBnClickedCornerDetection()
{
	// TODO: 在此加入控制項告知處理常式程式碼		
	String WindowName = "CDOutput";
	int i = 0;
	if (CCFunctionLIFlag == 0)//沒載入過就載入圖片
	{
		CCLoadingInput();
	}
	if(CCFunctionCDFlag==0)//沒偵測過就偵測角度
	{
		CCCornerDetection();
	}
	
	for (i = 0; i < 15; i++)
	{
		namedWindow(WindowName + to_string(i+1), WINDOW_NORMAL);
		imshow(WindowName + to_string(i + 1), CCInput[i]);
	}
	waitKey(0);
	destroyAllWindows();
	//system("CLS");
}


void COpencvHW2Dlg::OnBnClickedIntrinsicMatrix()//找內參
{
	// TODO: 在此加入控制項告知處理常式程式碼	
	if (CCFunctionLIFlag == 0)//沒載入過就載入圖片
	{
		CCLoadingInput();
	}
	if (CCFunctionCDFlag == 0)//沒偵測過就偵測角度
	{
		CCCornerDetection();
	}
	if (CCFunctionCalFlag == 0)//沒校準過就校準
	{
		CCCalibration();
	}
	cout << "--------Show Intrinsic Matrix--------" << endl;
	cout << CCIntrinsic << endl;
}


void COpencvHW2Dlg::OnBnClickedExtrinsicMatrix()//找外參
{
	// TODO: 在此加入控制項告知處理常式程式碼
	EMSelect = EMComboBox.GetCurSel();

	if (CCFunctionLIFlag == 0)//沒載入過就載入圖片
	{
		CCLoadingInput();
	}
	if (CCFunctionCDFlag == 0)//沒偵測過就偵測角度
	{
		CCCornerDetection();
	}
	if (CCFunctionCalFlag == 0)//沒校準過就校準
	{
		CCCalibration();
	}
	Mat Rotation,Translation=tvecs[EMSelect];
	Rodrigues(rvecs[EMSelect],Rotation);//將x,y,z方向旋轉量轉成旋轉矩陣
	hconcat(Rotation, Translation, Rotation);
	cout << "\n--------Show " << "Picture : " << "[ " << (EMSelect + 1) << " ]" << " Extrinsic Matrix--------" << endl;
	cout << Rotation<<endl;
}


void COpencvHW2Dlg::OnBnClickedDistortionMatrix()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	if (CCFunctionLIFlag == 0)//沒載入過就載入圖片
	{
		CCLoadingInput();
	}
	if (CCFunctionCDFlag == 0)//沒偵測過就偵測角度
	{
		CCCornerDetection();
	}
	if (CCFunctionCalFlag == 0)//沒校準過就校準
	{
		CCCalibration();
	}
	cout << "\n--------Show Distortion Coefficients--------" << endl;
	cout << CCDistCoeffs << endl;
}


void COpencvHW2Dlg::OnBnClickedAugmentedReality()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	int i = 0;
	if (CCFunctionLIFlag == 0)//沒載入過就載入圖片
	{
		CCLoadingInput();
	}
	if (CCFunctionCDFlag == 0)//沒偵測過就偵測角度
	{
		CCCornerDetection();
	}
	if (CCFunctionCalFlag == 0)//沒校準過就校準
	{
		CCCalibration();
	}
	vector<Point2f> ARProjectPoints;//儲存CUBE轉換完的該圖像2D座標
	vector<Point3f> ARCube = { Point3f(2,2,2),Point3f(2,0,2),Point3f(0,0,2),Point3f(0,2,2),
		                       Point3f(2,2,0),Point3f(2,0,0),Point3f(0,0,0),Point3f(0,2,0) };

	for (i = 0; i<5; i++)//將cube轉換從3D座標轉換到該圖的2D座標並將線畫出形成cube
	{
		projectPoints(ARCube, rvecs[i], tvecs[i], CCIntrinsic, CCDistCoeffs, ARProjectPoints);
		//算出cube的3D座標轉換為該2D圖的座標
		line(ARInput[i], ARProjectPoints[0], ARProjectPoints[1], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[0], ARProjectPoints[3], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[0], ARProjectPoints[4], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[1], ARProjectPoints[5], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[1], ARProjectPoints[2], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[2], ARProjectPoints[3], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[2], ARProjectPoints[6], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[3], ARProjectPoints[7], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[4], ARProjectPoints[5], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[4], ARProjectPoints[7], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[5], ARProjectPoints[6], Scalar(0, 0, 255), 7, CV_AA);
		line(ARInput[i], ARProjectPoints[6], ARProjectPoints[7], Scalar(0, 0, 255), 7, CV_AA);
		//將每個點連接起來並將線設成紅色邊的寬度為7，CV_AA為消除顯示器畫面線邊緣的凹凸鋸齒
	}
	namedWindow("Augmented Reality", WINDOW_NORMAL);//將原圖縮小尺寸並印出
	Mat flipImg;
	for (i = 0; i<5; i++) 
	{
		resizeWindow("Augmented Reality", ARInput[i].size[0] / 4, ARInput[i].size[1] / 4);
		flip(ARInput[i],flipImg,-1);//將圖片翻轉180度
		imshow("Augmented Reality", flipImg);
		waitKey(500);
	}
	waitKey(0);
	destroyAllWindows();
}
