
// OpencvHW2Dlg.h : ���Y��
//

#pragma once
#include "afxwin.h"


// COpencvHW2Dlg ��ܤ��
class COpencvHW2Dlg : public CDialogEx
{
// �غc
public:
	COpencvHW2Dlg(CWnd* pParent = NULL);	// �зǫغc�禡

// ��ܤ�����
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_OPENCVHW2_DIALOG };
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
	afx_msg void OnBnClickedOriginalHistogram();
	afx_msg void OnBnClickedEqualizedHistogram();
	afx_msg void OnBnClickedHoughCircle();
	afx_msg void OnBnClickedHueHistogram();
	afx_msg void OnBnClickedBackProjection();
	afx_msg void OnBnClickedCornerDetection();
	afx_msg void OnBnClickedIntrinsicMatrix();
	afx_msg void OnBnClickedExtrinsicMatrix();
	afx_msg void OnBnClickedDistortionMatrix();
	CComboBox EMComboBox;
	afx_msg void OnBnClickedAugmentedReality();
};
