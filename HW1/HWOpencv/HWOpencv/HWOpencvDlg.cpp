
// HWOpencvDlg.cpp : 實作檔
//

#include "stdafx.h"
#include "HWOpencv.h"
#include "HWOpencvDlg.h"
#include "afxdialogex.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "opencv2\opencv.hpp"
#include "stdio.h"
#include "math.h"

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


// CHWOpencvDlg 對話方塊



CHWOpencvDlg::CHWOpencvDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_HWOPENCV_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CHWOpencvDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CHWOpencvDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CHWOpencvDlg::OnBnClickedLoadAImage)
	ON_BN_CLICKED(IDC_BUTTON2, &CHWOpencvDlg::OnBnClickedColorConversion)
	ON_BN_CLICKED(IDC_BUTTON3, &CHWOpencvDlg::OnBnClickedImageFlipping)
	ON_BN_CLICKED(IDC_BUTTON4, &CHWOpencvDlg::OnBnClickedBlend)
	ON_BN_CLICKED(IDC_BUTTON5, &CHWOpencvDlg::OnBnClickedEdgeDetect)
	ON_BN_CLICKED(IDC_BUTTON6, &CHWOpencvDlg::OnBnClickedImagePyramids)
	ON_BN_CLICKED(IDC_BUTTON7, &CHWOpencvDlg::OnBnClickedGlobalThreshold)
	ON_BN_CLICKED(IDC_BUTTON8, &CHWOpencvDlg::OnBnClickedLocalThreshold)
	ON_BN_CLICKED(IDC_BUTTON9, &CHWOpencvDlg::OnBnClickedTrransformation)
	ON_BN_CLICKED(IDC_BUTTON10, &CHWOpencvDlg::OnBnClickedPerspectiveTransform)
END_MESSAGE_MAP()


// CHWOpencvDlg 訊息處理常式

BOOL CHWOpencvDlg::OnInitDialog()
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

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void CHWOpencvDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CHWOpencvDlg::OnPaint()
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
HCURSOR CHWOpencvDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CHWOpencvDlg::OnBnClickedLoadAImage()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat LIInput = imread("images\\dog.bmp", CV_LOAD_IMAGE_UNCHANGED);
	namedWindow("LoadImage", WINDOW_AUTOSIZE);
	imshow("LoadImage", LIInput);
	printf("hello\n");
	printf("The Size of Load Picture is :\n");
	printf("Height = %d\n", LIInput.cols);
	printf("Width  = %d\n", LIInput.rows);
	printf("channels = %d\n", LIInput.channels());
	printf("depth = %d\n", LIInput.depth());
	printf("(CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6)\n");
	cvWaitKey(0);
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedColorConversion()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat bgr[3];
	Mat temp;
	Mat CCInput = imread("images\\color.png", CV_LOAD_IMAGE_UNCHANGED);
	imshow("BeforeCC", CCInput);
	split(CCInput, bgr);//將三個顏色通道分到3個陣列中
	temp = bgr[2];
	bgr[2] = bgr[0];
	bgr[0] = bgr[1];
	bgr[1] = temp;
	merge(bgr, 3, CCInput);//將更改過後的三個通道值合併
	imshow("AfterCC", CCInput);
	waitKey(0);
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedImageFlipping()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat tempFlip;
	Mat IFInput = imread("images\\dog.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("BeforeFlip", IFInput);
	flip(IFInput, tempFlip, 1);// >0:沿Y軸翻轉，0:沿X軸翻轉 ，<0:X.Y軸同時翻轉
	imshow("AfterFlip", tempFlip);
	waitKey(0);
	destroyAllWindows();
}

Mat BInput, BInputFlip, BlendInputFlip;
int BlendsliderValue;
void onBlend(int, void*)
{
	double weightForImg2 = (double)BlendsliderValue / 200;
	double weightForImg = (1 - weightForImg2);//分配兩個圖的混合比重

	addWeighted(BInput, weightForImg, BInputFlip, weightForImg2, 0, BlendInputFlip);
	imshow("B-WindowTrackbar", BlendInputFlip);
}

void CHWOpencvDlg::OnBnClickedBlend()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	BInput = imread("images\\dog.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("B-BeforeFlip", BInput);
	flip(BInput, BInputFlip, 1);// >0:沿Y軸翻轉，0:沿X軸翻轉 ，<0:X.Y軸同時翻轉
	imshow("B-AfterFlip", BInputFlip);
	namedWindow("B-WindowTrackbar", WINDOW_AUTOSIZE);
	createTrackbar("B-Trackbar", "B-WindowTrackbar", &BlendsliderValue, 200, onBlend);
	onBlend(BlendsliderValue, 0);
	waitKey(0);
	destroyAllWindows();
	BlendsliderValue = 0;//回復原始狀態數值
}

#define PI 3.14159265
int EDThrSliderValue = 40;//控制Threshold trackbar
Mat EDGlobalVerHorMix, EDThreshHold;//存取該Mat到全域變數中，以利trackbar取用

void onEDThr(int, void*)//控制第2題Threshold trackbar
{
	threshold(EDGlobalVerHorMix, EDThreshHold, EDThrSliderValue, 255, THRESH_TOZERO);

	imshow("EDMagnitudeThr", EDThreshHold);
}
int EDDirSliderValue = 0;//控制direction trackbar
Mat EDGlobalDirection;//用來存取該Mat Direction到全域變數
Mat EDtemp, EDtempA;//用來控制輸出符合該direction的像素輸出
int x, y;
void onEDPixelDirection(int, void*)//控制第2題角度的trackbar
{
	EDtempA = EDtemp.clone();
	for (x = 0; x < EDtempA.rows; x++)
	{
		for (y = 0; y < EDtempA.cols; y++)
		{
			if (EDGlobalDirection.at<short>(x, y) < EDDirSliderValue - 10)//限制PIXEL上下10度
			{
				EDtempA.at<uchar>(x, y) = 0;//不在該角度的pixel設為0
			}
			if (EDGlobalDirection.at<short>(x, y) > EDDirSliderValue + 10)
			{
				EDtempA.at<uchar>(x, y) = 0;
			}
		}
	}
	imshow("EDPixelDirection", EDtempA);
}

void CHWOpencvDlg::OnBnClickedEdgeDetect()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	int i = 0, j = 0;
	Mat EDOriginal = imread("images\\M8.jpg", CV_LOAD_IMAGE_UNCHANGED);//原始圖檔載入
	Mat EDOriToGray;//將原始檔轉成灰階格式
	Mat EDGrayGau;//將灰階格式的圖作gaussion blur
	Mat EDPadding;//存放padding完的結果
	cvtColor(EDOriginal, EDOriToGray, CV_BGR2GRAY);//轉成灰階圖
	GaussianBlur(EDOriToGray, EDGrayGau, Size(3, 3), 0, 0);//作高斯SMOOTH
	imshow("ED-GrayByGaussian", EDGrayGau);//秀出圖片

	copyMakeBorder(EDGrayGau, EDPadding, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));//作Padding
	EDPadding.convertTo(EDPadding, CV_8UC1);
	Mat horSobel(EDGrayGau.rows, EDGrayGau.cols, CV_8SC1, Scalar(0));//儲存水平邊界捲積運算結果
	Mat	verSobel(EDGrayGau.rows, EDGrayGau.cols, CV_8SC1, Scalar(0));//儲存垂直邊界捲積運算結果
	Mat	showHorSobel(EDGrayGau.rows, EDGrayGau.cols, CV_8UC1, Scalar(0));//儲存上方水平運算完取絕對值的結果
	Mat	showVerSobel(EDGrayGau.rows, EDGrayGau.cols, CV_8UC1, Scalar(0));//儲存上方垂直運算完取絕對值的結果

	for (i = 1; i < EDPadding.rows - 1; i++)//作filter 捲積運算 
	{
		for (j = 1; j < EDPadding.cols - 1; j++)
		{
			horSobel.at<char>(i - 1, j - 1) = (1) * EDPadding.at<uchar>(i - 1, j - 1) +
											  (2) * EDPadding.at<uchar>(i - 1, j    ) +
											  (1) * EDPadding.at<uchar>(i - 1, j + 1) +
											  (0) * EDPadding.at<uchar>(i    , j - 1) +
											  (0) * EDPadding.at<uchar>(i    , j    ) +
											  (0) * EDPadding.at<uchar>(i    , j + 1) +
											 (-1) * EDPadding.at<uchar>(i + 1, j - 1) +
											 (-2) * EDPadding.at<uchar>(i + 1, j    ) +
											 (-1) * EDPadding.at<uchar>(i + 1, j + 1);
		}//結果為水平邊界Gy
	}
	for (i = 1; i < EDPadding.rows - 1; i++)
	{
		for (j = 1; j < EDPadding.cols - 1; j++)
		{
			verSobel.at<char>(i - 1, j - 1) = (-1) * EDPadding.at<uchar>(i - 1, j - 1) +
											   (0) * EDPadding.at<uchar>(i - 1, j    ) +
											   (1) * EDPadding.at<uchar>(i - 1, j + 1) +
											  (-2) * EDPadding.at<uchar>(i    , j - 1) +
											   (0) * EDPadding.at<uchar>(i    , j    ) +
											   (2) * EDPadding.at<uchar>(i    , j + 1) +
											  (-1) * EDPadding.at<uchar>(i + 1, j - 1) +
											   (0) * EDPadding.at<uchar>(i + 1, j    ) +
											   (1) * EDPadding.at<uchar>(i + 1, j + 1);
		}//結果為垂直邊界Gx
	}

	for (i = 0; i < horSobel.rows; i++)//將作完的值取絕對值
	{
		for (j = 0; j < horSobel.cols; j++)
		{
			showHorSobel.at<uchar>(i, j) = abs(horSobel.at<char>(i, j));
			showVerSobel.at<uchar>(i, j) = abs(verSobel.at<char>(i, j));
		}
	}
	imshow("horizontal edges", showHorSobel);//印出
	imshow("vertical edges", showVerSobel);//印出

	Mat EDVerHormix(horSobel.rows, horSobel.cols, CV_16UC1, Scalar(0));//用來存放垂直與水平的平方相加開根號的結果
	for (i = 0; i < EDVerHormix.rows; i++)//作平方相加開根號
	{
		for (j = 0; j < EDVerHormix.cols; j++)
		{
			EDVerHormix.at<ushort>(i, j) = sqrt(horSobel.at<char>(i, j)*horSobel.at<char>(i, j) + verSobel.at<char>(i, j)*verSobel.at<char>(i, j));
		}
	}
	normalize(EDVerHormix, EDVerHormix, 0, 255, NORM_MINMAX);//將值轉換成0-255
	EDVerHormix.convertTo(EDVerHormix, CV_8UC1);
	EDGlobalVerHorMix = EDVerHormix.clone();//讓trackbar可以利用的全域變數修改秀出的圖
	namedWindow("EDMagnitudeThr", WINDOW_AUTOSIZE);
	createTrackbar("Thr-Trackbar", "EDMagnitudeThr", &EDThrSliderValue, 255, onEDThr);
	onEDThr(EDThrSliderValue, 0);

	EDGlobalDirection.create(horSobel.rows, horSobel.cols, CV_16SC1);//儲存角度運算結果
	for (i = 0; i < EDGlobalDirection.rows; i++)//作角度運算及將結果範圍限制在0-360度
	{
		for (j = 0; j < EDGlobalDirection.cols; j++)
		{
			EDGlobalDirection.at<short>(i, j) = (atan2(verSobel.at<char>(i, j), horSobel.at<char>(i, j))) * 180 / PI;
			if (EDGlobalDirection.at<short>(i, j) < 0)
			{
				EDGlobalDirection.at<short>(i, j) = EDGlobalDirection.at<short>(i, j) + 360;
			}
		}
	}
	CString tempInputA;
	GetDlgItem(IDC_EDIT5)->GetWindowTextW(tempInputA);//從輸入讀取數字並轉換格式
	if (tempInputA.GetLength() > 0)//判斷是否有輸入
	{
		EDDirSliderValue = _ttoi(tempInputA);
	}
	EDtemp = EDVerHormix.clone();
	namedWindow("EDPixelDirection", WINDOW_AUTOSIZE);
	createTrackbar("Dir-Trackbar", "EDPixelDirection", &EDDirSliderValue, 360, onEDPixelDirection);
	onEDPixelDirection(EDDirSliderValue, 0);

	waitKey(0);
	EDThrSliderValue = 40;//結束時初始化，下次開啟時為原始的狀態
	EDDirSliderValue = 0;
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedImagePyramids()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	IplImage* pyrInput = cvLoadImage("images//pyramids_Gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* pyrGauHalf = cvCreateImage(cvSize(pyrInput->height / 2, pyrInput->width / 2), pyrInput->depth, pyrInput->nChannels);
	IplImage* pyrGauQuarter = cvCreateImage(cvSize(pyrGauHalf->height / 2, pyrGauHalf->width / 2), pyrGauHalf->depth, pyrGauHalf->nChannels);
	IplImage* pyrUpHalf = cvCreateImage(cvSize(pyrInput->height / 2, pyrInput->width / 2), pyrInput->depth, pyrInput->nChannels);
	IplImage* pyrUpFull = cvCreateImage(cvSize(pyrInput->height, pyrInput->width), pyrInput->depth, pyrInput->nChannels);
	cvPyrDown(pyrInput, pyrGauHalf);
	cvPyrDown(pyrGauHalf, pyrGauQuarter);
	cvShowImage("GaussianLevel1", pyrGauHalf);//印出Gaussian pyramids Leve 1 
	cvPyrUp(pyrGauQuarter, pyrUpHalf);
	cvPyrUp(pyrGauHalf, pyrUpFull);
	IplImage* pyrLapFull = cvCreateImage(cvSize(pyrInput->height, pyrInput->width), pyrInput->depth, pyrInput->nChannels);
	IplImage* pyrLapHalf = cvCreateImage(cvSize(pyrInput->height / 2, pyrInput->width / 2), pyrInput->depth, pyrInput->nChannels);
	cvSub(pyrInput, pyrUpFull, pyrLapFull);//原圖減掉pyrUp得Laplacian Level 0
	cvSub(pyrGauHalf, pyrUpHalf, pyrLapHalf);//Gaussian Level 1減掉 pyrUp得Lavlacian Level 1
	cvShowImage("Laplacian0", pyrLapFull);//印出Laplacian Level 0;
	IplImage* InversePyr0 = cvCreateImage(cvSize(pyrInput->height, pyrInput->width), pyrInput->depth, pyrInput->nChannels);
	IplImage* InversePyr1 = cvCreateImage(cvSize(pyrInput->height / 2, pyrInput->width / 2), pyrInput->depth, pyrInput->nChannels);
	cvAdd(pyrUpFull, pyrLapFull, InversePyr0);
	cvAdd(pyrUpHalf, pyrLapHalf, InversePyr1);//pyrUp加Laplacian得回原圖
	cvShowImage("InverseLevel0", InversePyr0);//印出inverse的Level 0 
	cvShowImage("InverseLevel1", InversePyr1);//印出inverse的Level 1 
	waitKey(0);
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedGlobalThreshold()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat GTInput = imread("images//QR.png", CV_LOAD_IMAGE_UNCHANGED);
	imshow("GTOriginal Image", GTInput);
	Mat AfterGThr;
	threshold(GTInput, AfterGThr, 80, 255, THRESH_BINARY);
	imshow("Global Threshold Image", AfterGThr);
	waitKey(0);
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedLocalThreshold()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat LTInput = imread("images//QR.png", CV_LOAD_IMAGE_UNCHANGED);
	imshow("GTOriginal Image", LTInput);
	Mat LTGrayImg;
	cvtColor(LTInput, LTGrayImg, CV_BGR2GRAY);
	Mat LTAfterThr;
	adaptiveThreshold(LTGrayImg, LTAfterThr, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 19, -1);
	imshow("Adaptive Threshold Image", LTAfterThr);
	waitKey(0);
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedTrransformation()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	CString tempInput;
	Mat RSTInput = imread("images//OriginalTransform.png", CV_LOAD_IMAGE_UNCHANGED);
	imshow("RSTInputImage", RSTInput);
	float scaleInput = 0;//存放輸入的比例
	int angleInput = 0;//輸入的角度
	float tempA = 130, tempB = 125, TxInput = 0, TyInput = 0;//原圖中心點及X位移Y位移
	Mat RSTOutputStep, RSTOutputFinal;

	GetDlgItem(IDC_EDIT1)->GetWindowTextW(tempInput);//從輸入讀取數字並轉換格式
	angleInput = _ttoi(tempInput);
	GetDlgItem(IDC_EDIT2)->GetWindowTextW(tempInput);
	scaleInput = (float)(_ttof(tempInput));
	GetDlgItem(IDC_EDIT3)->GetWindowTextW(tempInput);
	TxInput = (float)(_ttof(tempInput));
	GetDlgItem(IDC_EDIT4)->GetWindowTextW(tempInput);
	TyInput = (float)(_ttof(tempInput));

	Point2f SrcImg[3];
	SrcImg[0] = Point2f(0, 0);
	SrcImg[1] = Point2f(tempA, 0);
	SrcImg[2] = Point2f(0, tempB);
	Point2f DesImg[3];
	DesImg[0] = Point2f(TxInput, TyInput);
	DesImg[1] = Point2f(tempA + TxInput, TyInput);
	DesImg[2] = Point2f(TxInput, tempB + TyInput);
	Mat matStep = getAffineTransform(SrcImg, DesImg);//算出從原圖到新圖位移距離的仿射矩陣
	warpAffine(RSTInput, RSTOutputStep, matStep, RSTInput.size());

	Point center = (RSTOutputStep.rows / 2, RSTOutputStep.cols / 2);//作旋轉及比例變換
	Mat rot_mat = getRotationMatrix2D(center, angleInput, scaleInput);
	warpAffine(RSTOutputStep, RSTOutputFinal, rot_mat, RSTInput.size());
	imshow("RSTOutputImage2", RSTOutputFinal);
	waitKey(0);
	destroyAllWindows();
}

Point VertexFirst(-1, -1);
Point VertexSecond(-1, -1);
Point VertexThird(-1, -1);
Point VertexFourth(-1, -1);
int countForMouseClick = 0;
Mat PTInput;
Mat PTOutput;
void onMouse(int Event, int x, int y, int flags, void* param)
{
	if (countForMouseClick < 4 && Event == CV_EVENT_LBUTTONDOWN)
	{
		if (countForMouseClick == 0)//第一個選取的點
		{
			VertexFirst.x = x;
			VertexFirst.y = y;
			countForMouseClick++;
		}
		else if (countForMouseClick == 1)//第二個選取的點
		{
			VertexSecond.x = x;
			VertexSecond.y = y;
			countForMouseClick++;
		}
		else if (countForMouseClick == 2)//第三個選取的點
		{
			VertexThird.x = x;
			VertexThird.y = y;
			countForMouseClick++;
		}
		else if (countForMouseClick == 3)//第四個選取的點
		{
			VertexFourth.x = x;
			VertexFourth.y = y;
			countForMouseClick++;
		}
		if (countForMouseClick == 4)
		{
			Point2f PTSrcImg[4];//計算仿射矩陣所用，原本的點
			PTSrcImg[0] = Point2f(VertexFirst.x, VertexFirst.y);
			PTSrcImg[1] = Point2f(VertexSecond.x, VertexSecond.y);
			PTSrcImg[2] = Point2f(VertexThird.x, VertexThird.y);
			PTSrcImg[3] = Point2f(VertexFourth.x, VertexFourth.y);
			Point2f PTDesImg[4];//後來點的位置
			PTDesImg[0] = Point2f(20, 20);
			PTDesImg[1] = Point2f(450, 20);
			PTDesImg[2] = Point2f(450, 450);
			PTDesImg[3] = Point2f(20, 450);
			Mat PTptmat = getPerspectiveTransform(PTSrcImg, PTDesImg);
			warpPerspective(PTInput, PTOutput, PTptmat, PTOutput.size());//作仿射轉換
			Mat PTOutputROI = PTOutput(Rect(20, 20, 430, 430));//選取ROI區域，刪除掉後面的背景
			namedWindow("PTOutputROI", WINDOW_NORMAL);
			imshow("PTOutputROI", PTOutputROI);
			countForMouseClick = 0;
		}
	}
}

void CHWOpencvDlg::OnBnClickedPerspectiveTransform()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	PTInput = imread("images//OriginalPerspective.png", CV_LOAD_IMAGE_UNCHANGED);
	namedWindow("PTInput", WINDOW_NORMAL);
	imshow("PTInput", PTInput);
	setMouseCallback("PTInput", onMouse);
	waitKey(0);
	destroyAllWindows();
}
