
// HWOpencvDlg.cpp : ��@��
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


// CHWOpencvDlg ��ܤ��



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


// CHWOpencvDlg �T���B�z�`��

BOOL CHWOpencvDlg::OnInitDialog()
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

	return TRUE;  // �Ǧ^ TRUE�A���D�z�ﱱ��]�w�J�I
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

// �p�G�N�̤p�ƫ��s�[�J�z����ܤ���A�z�ݭn�U�C���{���X�A
// �H�Kø�s�ϥܡC���ϥΤ��/�˵��Ҧ��� MFC ���ε{���A
// �ج[�|�۰ʧ������@�~�C

void CHWOpencvDlg::OnPaint()
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
HCURSOR CHWOpencvDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CHWOpencvDlg::OnBnClickedLoadAImage()
{
	// TODO: �b���[�J����i���B�z�`���{���X
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
	// TODO: �b���[�J����i���B�z�`���{���X
	Mat bgr[3];
	Mat temp;
	Mat CCInput = imread("images\\color.png", CV_LOAD_IMAGE_UNCHANGED);
	imshow("BeforeCC", CCInput);
	split(CCInput, bgr);//�N�T���C��q�D����3�Ӱ}�C��
	temp = bgr[2];
	bgr[2] = bgr[0];
	bgr[0] = bgr[1];
	bgr[1] = temp;
	merge(bgr, 3, CCInput);//�N���L�᪺�T�ӳq�D�ȦX��
	imshow("AfterCC", CCInput);
	waitKey(0);
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedImageFlipping()
{
	// TODO: �b���[�J����i���B�z�`���{���X
	Mat tempFlip;
	Mat IFInput = imread("images\\dog.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("BeforeFlip", IFInput);
	flip(IFInput, tempFlip, 1);// >0:�uY�b½��A0:�uX�b½�� �A<0:X.Y�b�P��½��
	imshow("AfterFlip", tempFlip);
	waitKey(0);
	destroyAllWindows();
}

Mat BInput, BInputFlip, BlendInputFlip;
int BlendsliderValue;
void onBlend(int, void*)
{
	double weightForImg2 = (double)BlendsliderValue / 200;
	double weightForImg = (1 - weightForImg2);//���t��ӹϪ��V�X��

	addWeighted(BInput, weightForImg, BInputFlip, weightForImg2, 0, BlendInputFlip);
	imshow("B-WindowTrackbar", BlendInputFlip);
}

void CHWOpencvDlg::OnBnClickedBlend()
{
	// TODO: �b���[�J����i���B�z�`���{���X
	BInput = imread("images\\dog.bmp", CV_LOAD_IMAGE_UNCHANGED);
	imshow("B-BeforeFlip", BInput);
	flip(BInput, BInputFlip, 1);// >0:�uY�b½��A0:�uX�b½�� �A<0:X.Y�b�P��½��
	imshow("B-AfterFlip", BInputFlip);
	namedWindow("B-WindowTrackbar", WINDOW_AUTOSIZE);
	createTrackbar("B-Trackbar", "B-WindowTrackbar", &BlendsliderValue, 200, onBlend);
	onBlend(BlendsliderValue, 0);
	waitKey(0);
	destroyAllWindows();
	BlendsliderValue = 0;//�^�_��l���A�ƭ�
}

#define PI 3.14159265
int EDThrSliderValue = 40;//����Threshold trackbar
Mat EDGlobalVerHorMix, EDThreshHold;//�s����Mat������ܼƤ��A�H�Qtrackbar����

void onEDThr(int, void*)//�����2�DThreshold trackbar
{
	threshold(EDGlobalVerHorMix, EDThreshHold, EDThrSliderValue, 255, THRESH_TOZERO);

	imshow("EDMagnitudeThr", EDThreshHold);
}
int EDDirSliderValue = 0;//����direction trackbar
Mat EDGlobalDirection;//�ΨӦs����Mat Direction������ܼ�
Mat EDtemp, EDtempA;//�Ψӱ����X�ŦX��direction��������X
int x, y;
void onEDPixelDirection(int, void*)//�����2�D���ת�trackbar
{
	EDtempA = EDtemp.clone();
	for (x = 0; x < EDtempA.rows; x++)
	{
		for (y = 0; y < EDtempA.cols; y++)
		{
			if (EDGlobalDirection.at<short>(x, y) < EDDirSliderValue - 10)//����PIXEL�W�U10��
			{
				EDtempA.at<uchar>(x, y) = 0;//���b�Ө��ת�pixel�]��0
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
	// TODO: �b���[�J����i���B�z�`���{���X
	int i = 0, j = 0;
	Mat EDOriginal = imread("images\\M8.jpg", CV_LOAD_IMAGE_UNCHANGED);//��l���ɸ��J
	Mat EDOriToGray;//�N��l���ন�Ƕ��榡
	Mat EDGrayGau;//�N�Ƕ��榡���ϧ@gaussion blur
	Mat EDPadding;//�s��padding�������G
	cvtColor(EDOriginal, EDOriToGray, CV_BGR2GRAY);//�ন�Ƕ���
	GaussianBlur(EDOriToGray, EDGrayGau, Size(3, 3), 0, 0);//�@����SMOOTH
	imshow("ED-GrayByGaussian", EDGrayGau);//�q�X�Ϥ�

	copyMakeBorder(EDGrayGau, EDPadding, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));//�@Padding
	EDPadding.convertTo(EDPadding, CV_8UC1);
	Mat horSobel(EDGrayGau.rows, EDGrayGau.cols, CV_8SC1, Scalar(0));//�x�s������ɱ��n�B�⵲�G
	Mat	verSobel(EDGrayGau.rows, EDGrayGau.cols, CV_8SC1, Scalar(0));//�x�s������ɱ��n�B�⵲�G
	Mat	showHorSobel(EDGrayGau.rows, EDGrayGau.cols, CV_8UC1, Scalar(0));//�x�s�W������B�⧹������Ȫ����G
	Mat	showVerSobel(EDGrayGau.rows, EDGrayGau.cols, CV_8UC1, Scalar(0));//�x�s�W�諫���B�⧹������Ȫ����G

	for (i = 1; i < EDPadding.rows - 1; i++)//�@filter ���n�B�� 
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
		}//���G���������Gy
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
		}//���G���������Gx
	}

	for (i = 0; i < horSobel.rows; i++)//�N�@�����Ȩ������
	{
		for (j = 0; j < horSobel.cols; j++)
		{
			showHorSobel.at<uchar>(i, j) = abs(horSobel.at<char>(i, j));
			showVerSobel.at<uchar>(i, j) = abs(verSobel.at<char>(i, j));
		}
	}
	imshow("horizontal edges", showHorSobel);//�L�X
	imshow("vertical edges", showVerSobel);//�L�X

	Mat EDVerHormix(horSobel.rows, horSobel.cols, CV_16UC1, Scalar(0));//�ΨӦs�񫫪��P����������ۥ[�}�ڸ������G
	for (i = 0; i < EDVerHormix.rows; i++)//�@����ۥ[�}�ڸ�
	{
		for (j = 0; j < EDVerHormix.cols; j++)
		{
			EDVerHormix.at<ushort>(i, j) = sqrt(horSobel.at<char>(i, j)*horSobel.at<char>(i, j) + verSobel.at<char>(i, j)*verSobel.at<char>(i, j));
		}
	}
	normalize(EDVerHormix, EDVerHormix, 0, 255, NORM_MINMAX);//�N���ഫ��0-255
	EDVerHormix.convertTo(EDVerHormix, CV_8UC1);
	EDGlobalVerHorMix = EDVerHormix.clone();//��trackbar�i�H�Q�Ϊ������ܼƭק�q�X����
	namedWindow("EDMagnitudeThr", WINDOW_AUTOSIZE);
	createTrackbar("Thr-Trackbar", "EDMagnitudeThr", &EDThrSliderValue, 255, onEDThr);
	onEDThr(EDThrSliderValue, 0);

	EDGlobalDirection.create(horSobel.rows, horSobel.cols, CV_16SC1);//�x�s���׹B�⵲�G
	for (i = 0; i < EDGlobalDirection.rows; i++)//�@���׹B��αN���G�d�򭭨�b0-360��
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
	GetDlgItem(IDC_EDIT5)->GetWindowTextW(tempInputA);//�q��JŪ���Ʀr���ഫ�榡
	if (tempInputA.GetLength() > 0)//�P�_�O�_����J
	{
		EDDirSliderValue = _ttoi(tempInputA);
	}
	EDtemp = EDVerHormix.clone();
	namedWindow("EDPixelDirection", WINDOW_AUTOSIZE);
	createTrackbar("Dir-Trackbar", "EDPixelDirection", &EDDirSliderValue, 360, onEDPixelDirection);
	onEDPixelDirection(EDDirSliderValue, 0);

	waitKey(0);
	EDThrSliderValue = 40;//�����ɪ�l�ơA�U���}�Үɬ���l�����A
	EDDirSliderValue = 0;
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedImagePyramids()
{
	// TODO: �b���[�J����i���B�z�`���{���X
	IplImage* pyrInput = cvLoadImage("images//pyramids_Gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* pyrGauHalf = cvCreateImage(cvSize(pyrInput->height / 2, pyrInput->width / 2), pyrInput->depth, pyrInput->nChannels);
	IplImage* pyrGauQuarter = cvCreateImage(cvSize(pyrGauHalf->height / 2, pyrGauHalf->width / 2), pyrGauHalf->depth, pyrGauHalf->nChannels);
	IplImage* pyrUpHalf = cvCreateImage(cvSize(pyrInput->height / 2, pyrInput->width / 2), pyrInput->depth, pyrInput->nChannels);
	IplImage* pyrUpFull = cvCreateImage(cvSize(pyrInput->height, pyrInput->width), pyrInput->depth, pyrInput->nChannels);
	cvPyrDown(pyrInput, pyrGauHalf);
	cvPyrDown(pyrGauHalf, pyrGauQuarter);
	cvShowImage("GaussianLevel1", pyrGauHalf);//�L�XGaussian pyramids Leve 1 
	cvPyrUp(pyrGauQuarter, pyrUpHalf);
	cvPyrUp(pyrGauHalf, pyrUpFull);
	IplImage* pyrLapFull = cvCreateImage(cvSize(pyrInput->height, pyrInput->width), pyrInput->depth, pyrInput->nChannels);
	IplImage* pyrLapHalf = cvCreateImage(cvSize(pyrInput->height / 2, pyrInput->width / 2), pyrInput->depth, pyrInput->nChannels);
	cvSub(pyrInput, pyrUpFull, pyrLapFull);//��ϴpyrUp�oLaplacian Level 0
	cvSub(pyrGauHalf, pyrUpHalf, pyrLapHalf);//Gaussian Level 1� pyrUp�oLavlacian Level 1
	cvShowImage("Laplacian0", pyrLapFull);//�L�XLaplacian Level 0;
	IplImage* InversePyr0 = cvCreateImage(cvSize(pyrInput->height, pyrInput->width), pyrInput->depth, pyrInput->nChannels);
	IplImage* InversePyr1 = cvCreateImage(cvSize(pyrInput->height / 2, pyrInput->width / 2), pyrInput->depth, pyrInput->nChannels);
	cvAdd(pyrUpFull, pyrLapFull, InversePyr0);
	cvAdd(pyrUpHalf, pyrLapHalf, InversePyr1);//pyrUp�[Laplacian�o�^���
	cvShowImage("InverseLevel0", InversePyr0);//�L�Xinverse��Level 0 
	cvShowImage("InverseLevel1", InversePyr1);//�L�Xinverse��Level 1 
	waitKey(0);
	destroyAllWindows();
}


void CHWOpencvDlg::OnBnClickedGlobalThreshold()
{
	// TODO: �b���[�J����i���B�z�`���{���X
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
	// TODO: �b���[�J����i���B�z�`���{���X
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
	// TODO: �b���[�J����i���B�z�`���{���X
	CString tempInput;
	Mat RSTInput = imread("images//OriginalTransform.png", CV_LOAD_IMAGE_UNCHANGED);
	imshow("RSTInputImage", RSTInput);
	float scaleInput = 0;//�s���J�����
	int angleInput = 0;//��J������
	float tempA = 130, tempB = 125, TxInput = 0, TyInput = 0;//��Ϥ����I��X�첾Y�첾
	Mat RSTOutputStep, RSTOutputFinal;

	GetDlgItem(IDC_EDIT1)->GetWindowTextW(tempInput);//�q��JŪ���Ʀr���ഫ�榡
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
	Mat matStep = getAffineTransform(SrcImg, DesImg);//��X�q��Ϩ�s�Ϧ첾�Z������g�x�}
	warpAffine(RSTInput, RSTOutputStep, matStep, RSTInput.size());

	Point center = (RSTOutputStep.rows / 2, RSTOutputStep.cols / 2);//�@����Τ���ܴ�
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
		if (countForMouseClick == 0)//�Ĥ@�ӿ�����I
		{
			VertexFirst.x = x;
			VertexFirst.y = y;
			countForMouseClick++;
		}
		else if (countForMouseClick == 1)//�ĤG�ӿ�����I
		{
			VertexSecond.x = x;
			VertexSecond.y = y;
			countForMouseClick++;
		}
		else if (countForMouseClick == 2)//�ĤT�ӿ�����I
		{
			VertexThird.x = x;
			VertexThird.y = y;
			countForMouseClick++;
		}
		else if (countForMouseClick == 3)//�ĥ|�ӿ�����I
		{
			VertexFourth.x = x;
			VertexFourth.y = y;
			countForMouseClick++;
		}
		if (countForMouseClick == 4)
		{
			Point2f PTSrcImg[4];//�p���g�x�}�ҥΡA�쥻���I
			PTSrcImg[0] = Point2f(VertexFirst.x, VertexFirst.y);
			PTSrcImg[1] = Point2f(VertexSecond.x, VertexSecond.y);
			PTSrcImg[2] = Point2f(VertexThird.x, VertexThird.y);
			PTSrcImg[3] = Point2f(VertexFourth.x, VertexFourth.y);
			Point2f PTDesImg[4];//����I����m
			PTDesImg[0] = Point2f(20, 20);
			PTDesImg[1] = Point2f(450, 20);
			PTDesImg[2] = Point2f(450, 450);
			PTDesImg[3] = Point2f(20, 450);
			Mat PTptmat = getPerspectiveTransform(PTSrcImg, PTDesImg);
			warpPerspective(PTInput, PTOutput, PTptmat, PTOutput.size());//�@��g�ഫ
			Mat PTOutputROI = PTOutput(Rect(20, 20, 430, 430));//���ROI�ϰ�A�R�����᭱���I��
			namedWindow("PTOutputROI", WINDOW_NORMAL);
			imshow("PTOutputROI", PTOutputROI);
			countForMouseClick = 0;
		}
	}
}

void CHWOpencvDlg::OnBnClickedPerspectiveTransform()
{
	// TODO: �b���[�J����i���B�z�`���{���X
	PTInput = imread("images//OriginalPerspective.png", CV_LOAD_IMAGE_UNCHANGED);
	namedWindow("PTInput", WINDOW_NORMAL);
	imshow("PTInput", PTInput);
	setMouseCallback("PTInput", onMouse);
	waitKey(0);
	destroyAllWindows();
}
