
// HWOpencvDlg.h : ���Y��
//

#pragma once


// CHWOpencvDlg ��ܤ��
class CHWOpencvDlg : public CDialogEx
{
// �غc
public:
	CHWOpencvDlg(CWnd* pParent = NULL);	// �зǫغc�禡

// ��ܤ�����
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_HWOPENCV_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �䴩


// �{���X��@
protected:
	HICON m_hIcon;

	// ���ͪ��T�������禡
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedLoadAImage();
	afx_msg void OnBnClickedColorConversion();
	afx_msg void OnBnClickedImageFlipping();
	afx_msg void OnBnClickedBlend();
	afx_msg void OnBnClickedEdgeDetect();
	afx_msg void OnBnClickedImagePyramids();
	afx_msg void OnBnClickedGlobalThreshold();
	afx_msg void OnBnClickedLocalThreshold();
	afx_msg void OnBnClickedTrransformation();
	afx_msg void OnBnClickedPerspectiveTransform();
};
