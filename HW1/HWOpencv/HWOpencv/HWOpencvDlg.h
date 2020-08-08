
// HWOpencvDlg.h : 標頭檔
//

#pragma once


// CHWOpencvDlg 對話方塊
class CHWOpencvDlg : public CDialogEx
{
// 建構
public:
	CHWOpencvDlg(CWnd* pParent = NULL);	// 標準建構函式

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_HWOPENCV_DIALOG };
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
