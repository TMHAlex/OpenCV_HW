
// OpencvHW2Dlg.h : 標頭檔
//

#pragma once
#include "afxwin.h"


// COpencvHW2Dlg 對話方塊
class COpencvHW2Dlg : public CDialogEx
{
// 建構
public:
	COpencvHW2Dlg(CWnd* pParent = NULL);	// 標準建構函式

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_OPENCVHW2_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支援


// 程式碼實作
protected:
	HICON m_hIcon;

	// 產生的訊息對應函式
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
