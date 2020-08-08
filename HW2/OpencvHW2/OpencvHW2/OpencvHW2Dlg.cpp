
// OpencvHW2Dlg.cpp : ��@��
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


// �� App About �ϥ� CAboutDlg ��ܤ��

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// ��ܤ�����
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �䴩

// �{���X��@
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


// COpencvHW2Dlg ��ܤ��



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


// COpencvHW2Dlg �T���B�z�`��

BOOL COpencvHW2Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �N [����...] �\���[�J�t�Υ\���C

	// IDM_ABOUTBOX �����b�t�ΩR�O�d�򤧤��C
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

	// �]�w����ܤ�����ϥܡC�����ε{�����D�������O��ܤ���ɡA
	// �ج[�|�۰ʱq�Ʀ��@�~
	SetIcon(m_hIcon, TRUE);			// �]�w�j�ϥ�
	SetIcon(m_hIcon, FALSE);		// �]�w�p�ϥ�

	// TODO: �b���[�J�B�~����l�]�w
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	CString numInComboBox;
	numInComboBox.Empty();
	for (int i = 0; i < 15; i++)
	{
		numInComboBox.Format(_T("Picture : %d"), i + 1);
		EMComboBox.AddString(numInComboBox);
	}

	return TRUE;  // �Ǧ^ TRUE�A���D�z�ﱱ��]�w�J�I
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

// �p�G�N�̤p�ƫ��s�[�J�z����ܤ���A�z�ݭn�U�C���{���X�A
// �H�Kø�s�ϥܡC���ϥΤ��/�˵��Ҧ��� MFC ���ε{���A
// �ج[�|�۰ʧ������@�~�C

void COpencvHW2Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ø�s���˸m���e

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// �N�ϥܸm����Τ�ݯx��
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �yø�ϥ�
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ��ϥΪ̩즲�̤p�Ƶ����ɡA
// �t�ΩI�s�o�ӥ\����o�����ܡC
HCURSOR COpencvHW2Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void COpencvHW2Dlg::OnBnClickedOriginalHistogram()//1-1�e�X�����
{
	// TODO: �b���[�J����i���B�z�`���{���X
	Mat OHinput = imread("images\\plant.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("OHinputphoto", OHinput);
	int histSize = 256;//�����bin��(��b)
	float range[] = { 0, 255 };//�έp���d��
	const float* histRange = { range };
	Mat histImg;
	Mat OHshowHistImg(256, 256, CV_8UC1, Scalar(255));  //�s��έp�ƾڪ�����ϼv��
	calcHist(&OHinput, 1, 0, Mat(), histImg, 1, &histSize, &histRange);//�έp�Ϥ����
	float histMaxValue = 0;//�s��έp�Ƕ��ƾڪ��̤j��
	for (int i = 0; i<histSize; i++)//��X�̤j��
	{
		float tempValue = histImg.at<float>(i);
		if (histMaxValue < tempValue) {
			histMaxValue = tempValue;
		}
	}

	float scale = (0.9 * 256) / histMaxValue;//�Ϥ��Y����
	for (int i = 0; i<histSize; i++)//�e�u
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
	// TODO: �b���[�J����i���B�z�`���{���X
	Mat EHinput = imread("images\\plant.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat EHoutput;
	equalizeHist(EHinput, EHoutput);//�N�Ϥ���T�@����
	imshow("EHOutput", EHoutput);
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	Mat histImg;
	calcHist(&EHoutput, 1, 0, Mat(), histImg, 1, &histSize, &histRange);

	Mat EHshowHistImg(256, 256, CV_8UC1, Scalar(255));  //�s��έp�ƾڪ�����ϼv��
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
	// TODO: �b���[�J����i���B�z�`���{���X
	Mat HCInput = imread("images\\q2_train.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat HCOutput;
	HCOutput = HCInput.clone();
	vector<Vec3f> HCCircles;//�x�s��ߥb�|
	Mat HCsrcGray;
	cvtColor(HCInput, HCsrcGray, CV_BGR2GRAY);
	GaussianBlur(HCsrcGray, HCsrcGray, Size(9, 9), 2, 2);//����Ϲ��@�U�ƳB�z�h�����T
	HoughCircles(HCsrcGray, HCCircles, CV_HOUGH_GRADIENT, 1.5, 30, 50, 32, 15, 20);//���
	for (int i = 0; i < HCCircles.size(); i++)//�N�Ϲ�����ΰ�_��
	{
		Point center(cvRound(HCCircles[i][0]), cvRound(HCCircles[i][1]));
		int radius = cvRound(HCCircles[i][2]);
		//cout << cvRound(HCCircles[i][0]) << "\t" << cvRound(HCCircles[i][1]) << "\t" << cvRound(HCCircles[i][2]) << endl;
		circle(HCOutput, center, radius, Scalar(0, 255, 0), 2, 8, 0);//�e��(�ϡA���ߡA�b�|�A�C��A��ɤj�p�A8�s�q)
		HCOutput.at<Vec3b>(center)[0] = 0;//���ߵ۬���
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
	// TODO: �b���[�J����i���B�z�`���{���X
	Mat HHInput = imread("images\\q2_train.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat HHsrcGray;
	cvtColor(HHInput, HHsrcGray, CV_BGR2GRAY);
	Mat HHCirclesImg(HHInput.rows, HHInput.cols, CV_8UC3, Scalar(0));
	Mat HHCirclesHSVImg;
	vector<Vec3f> HHCircles;//�x�s��쪺��ߤΥb�|
	GaussianBlur(HHsrcGray, HHsrcGray, Size(9, 9), 2, 2);
	//1.5, 15, 50, 24, 8, 12   test�ϰѼ�
	//1.5, 30, 50, 32, 15, 20 train�ϰѼ�
	HoughCircles(HHsrcGray, HHCircles, CV_HOUGH_GRADIENT, 1.5, 30, 50, 32, 15, 20);
	for (int i = 0; i < HHCircles.size(); i++)
	{
		Point center(cvRound(HHCircles[i][0]), cvRound(HHCircles[i][1]));
		int radius = cvRound(HHCircles[i][2]);
		circle(HHCirclesImg, center, radius, Scalar(HHInput.at<Vec3b>(center)[0], HHInput.at<Vec3b>(center)[1], HHInput.at<Vec3b>(center)[2]), CV_FILLED, 8, 0);
	    //�N��쪺��s��@�i�s���Ϲ����A�s���Ϥ��u�O�s�o�Ƕ��
	}
	imshow("HHInput", HHInput);
	//imshow("HHCirclesTest",HHCirclesImg);

	cvtColor(HHCirclesImg, HHCirclesHSVImg, CV_BGR2HSV);
	int histSize = 256;
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };
	Mat histImg;//�x�s�έp���G
	Mat HHshowHistImg(256, 256, CV_8UC1, Scalar(255));  //�N�έp���G�ഫ���@�i�����
	calcHist(&HHCirclesHSVImg, 1, channels, Mat(), histImg, 1, &histSize, ranges);
	float histMaxValue = 0;
	for (int i = 1; i<histSize; i++)//�έp�̤j��
	{
		float tempValue = histImg.at<float>(i);
		if (histMaxValue < tempValue) {
			histMaxValue = tempValue;
		}
	}

	float scale = (0.9 * 256) / histMaxValue;
	for (int i = 1; i<histSize; i++)//�e��
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
	// TODO: �b���[�J����i���B�z�`���{���X
	Mat BPtrainInput = imread("images\\q2_train.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat BPtestInput= imread("images\\q2_test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat BPtestHSV;
	cvtColor(BPtestInput, BPtestHSV, CV_BGR2HSV);
	Mat BPsrcGray;//�N�Ϥ�train �ন�Ƕ��ϥΨӰ�HoughCirclr
	cvtColor(BPtrainInput, BPsrcGray, CV_BGR2GRAY);
	Mat BPCirclesImg(BPtrainInput.rows, BPtrainInput.cols, CV_8UC3, Scalar(0));
	Mat BPCirclesHSVImg;
	vector<Vec3f> BPCircles;
	GaussianBlur(BPsrcGray, BPsrcGray, Size(9, 9), 2, 2);
	HoughCircles(BPsrcGray, BPCircles, CV_HOUGH_GRADIENT, 1.5, 30, 50, 32, 15, 20);
	for (int i = 0; i < BPCircles.size(); i++)//�N��쪺�꭫�s�e�b�@�ӷs���Ϥ���
	{
		Point center(cvRound(BPCircles[i][0]), cvRound(BPCircles[i][1]));
		int radius = cvRound(BPCircles[i][2]);
		circle(BPCirclesImg, center, radius, Scalar(BPtrainInput.at<Vec3b>(center)[0], BPtrainInput.at<Vec3b>(center)[1], BPtrainInput.at<Vec3b>(center)[2]), CV_FILLED, 8, 0);
	}
	
	//�N��쪺����x�s�Ϥ��ഫ��HSV��
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
	//��HSV�ϧ@��Histogram�óW�@��
	calcBackProject(&BPtestHSV, 1, HSVChannel, BPHistImg, srcBackProject, BackProjectRange, 255, true);
	//normalize(srcBackProject, srcBackProject, 0, 255, NORM_MINMAX);
	//�Q�γW�@�ƪ��ƾک�J�n�������Ϥ����A�P�_�Ϥ��O�_�ŦX������ϡA�X�Ӫ��ƾڬ����v�Ȧ��0-1����
	//�p�G�N�����v�ƾڳW�@��0-255�A�N��զⳡ���V�ŦX�������
	imshow("BPtestInput", BPtestInput);
	imshow("BackProjection_Result", srcBackProject);
	waitKey(0);
	destroyAllWindows();
}

Mat CCInput[15];//�s��15�i�Ϥ�
Size cornerSize = Size(11, 8);//�ѽL��j�p
vector<Point2f> CCcorners;//�x�s��쪺corner
bool CCDetectResult[15];//�x�scorner�������G �Ȭ�0��1
bool CCFunctionLIFlag = 0;//������J�Ϥ����X��
bool CCFunctionCDFlag = 0;//����װ������X��
bool CCFunctionCalFlag = 0;//����շǪ��X��

vector<vector<Point2f>> IMImagePoint;
vector<vector<Point3f>> IMObjectPoint;
Mat CCIntrinsic = Mat(3, 3, CV_32FC1);//����
Mat CCDistCoeffs;//�x�s�]���z��βո˧޳N�~�t���ե��Ѽ�
vector<Mat> rvecs;//�x�sRotation�Ѽ�
vector<Mat> tvecs;//�x�stranslation�Ѽ�
int EMSelect = 0;//�x�s2-3�ҿ�ܪ��Ϥ�

Mat ARInput[5];

void CCLoadingInput()//���J�Ϥ�
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

void CCCornerDetection()//���װ���
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

void CCCalibration()//�շ�
{
	int numCors = cornerSize.width*cornerSize.height;
	vector<Point3f> chessborad_pts;
	int i = 0;
	cout << "Start Calibration..." << endl;
	for (i = 0; i<numCors; i++)
	{
		chessborad_pts.push_back(Point3f(i / cornerSize.width, i%cornerSize.width, 0.0f));//�إ�3D�I�y��
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
	// TODO: �b���[�J����i���B�z�`���{���X		
	String WindowName = "CDOutput";
	int i = 0;
	if (CCFunctionLIFlag == 0)//�S���J�L�N���J�Ϥ�
	{
		CCLoadingInput();
	}
	if(CCFunctionCDFlag==0)//�S�����L�N��������
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


void COpencvHW2Dlg::OnBnClickedIntrinsicMatrix()//�䤺��
{
	// TODO: �b���[�J����i���B�z�`���{���X	
	if (CCFunctionLIFlag == 0)//�S���J�L�N���J�Ϥ�
	{
		CCLoadingInput();
	}
	if (CCFunctionCDFlag == 0)//�S�����L�N��������
	{
		CCCornerDetection();
	}
	if (CCFunctionCalFlag == 0)//�S�շǹL�N�շ�
	{
		CCCalibration();
	}
	cout << "--------Show Intrinsic Matrix--------" << endl;
	cout << CCIntrinsic << endl;
}


void COpencvHW2Dlg::OnBnClickedExtrinsicMatrix()//��~��
{
	// TODO: �b���[�J����i���B�z�`���{���X
	EMSelect = EMComboBox.GetCurSel();

	if (CCFunctionLIFlag == 0)//�S���J�L�N���J�Ϥ�
	{
		CCLoadingInput();
	}
	if (CCFunctionCDFlag == 0)//�S�����L�N��������
	{
		CCCornerDetection();
	}
	if (CCFunctionCalFlag == 0)//�S�շǹL�N�շ�
	{
		CCCalibration();
	}
	Mat Rotation,Translation=tvecs[EMSelect];
	Rodrigues(rvecs[EMSelect],Rotation);//�Nx,y,z��V����q�ন����x�}
	hconcat(Rotation, Translation, Rotation);
	cout << "\n--------Show " << "Picture : " << "[ " << (EMSelect + 1) << " ]" << " Extrinsic Matrix--------" << endl;
	cout << Rotation<<endl;
}


void COpencvHW2Dlg::OnBnClickedDistortionMatrix()
{
	// TODO: �b���[�J����i���B�z�`���{���X
	if (CCFunctionLIFlag == 0)//�S���J�L�N���J�Ϥ�
	{
		CCLoadingInput();
	}
	if (CCFunctionCDFlag == 0)//�S�����L�N��������
	{
		CCCornerDetection();
	}
	if (CCFunctionCalFlag == 0)//�S�շǹL�N�շ�
	{
		CCCalibration();
	}
	cout << "\n--------Show Distortion Coefficients--------" << endl;
	cout << CCDistCoeffs << endl;
}


void COpencvHW2Dlg::OnBnClickedAugmentedReality()
{
	// TODO: �b���[�J����i���B�z�`���{���X
	int i = 0;
	if (CCFunctionLIFlag == 0)//�S���J�L�N���J�Ϥ�
	{
		CCLoadingInput();
	}
	if (CCFunctionCDFlag == 0)//�S�����L�N��������
	{
		CCCornerDetection();
	}
	if (CCFunctionCalFlag == 0)//�S�շǹL�N�շ�
	{
		CCCalibration();
	}
	vector<Point2f> ARProjectPoints;//�x�sCUBE�ഫ�����ӹϹ�2D�y��
	vector<Point3f> ARCube = { Point3f(2,2,2),Point3f(2,0,2),Point3f(0,0,2),Point3f(0,2,2),
		                       Point3f(2,2,0),Point3f(2,0,0),Point3f(0,0,0),Point3f(0,2,0) };

	for (i = 0; i<5; i++)//�Ncube�ഫ�q3D�y���ഫ��ӹϪ�2D�y�ШñN�u�e�X�Φ�cube
	{
		projectPoints(ARCube, rvecs[i], tvecs[i], CCIntrinsic, CCDistCoeffs, ARProjectPoints);
		//��Xcube��3D�y���ഫ����2D�Ϫ��y��
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
		//�N�C���I�s���_�ӨñN�u�]�������䪺�e�׬�7�ACV_AA��������ܾ��e���u��t���W�Y����
	}
	namedWindow("Augmented Reality", WINDOW_NORMAL);//�N����Y�p�ؤo�æL�X
	Mat flipImg;
	for (i = 0; i<5; i++) 
	{
		resizeWindow("Augmented Reality", ARInput[i].size[0] / 4, ARInput[i].size[1] / 4);
		flip(ARInput[i],flipImg,-1);//�N�Ϥ�½��180��
		imshow("Augmented Reality", flipImg);
		waitKey(500);
	}
	waitKey(0);
	destroyAllWindows();
}
