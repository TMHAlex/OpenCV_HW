
// HWOpencv.h : PROJECT_NAME 應用程式的主要標頭檔
//

#pragma once

#ifndef __AFXWIN_H__
	#error "對 PCH 包含此檔案前先包含 'stdafx.h'"
#endif

#include "resource.h"		// 主要符號


// CHWOpencvApp: 
// 請參閱實作此類別的 HWOpencv.cpp
//

class CHWOpencvApp : public CWinApp
{
public:
	CHWOpencvApp();

// 覆寫
public:
	virtual BOOL InitInstance();

// 程式碼實作

	DECLARE_MESSAGE_MAP()
};

extern CHWOpencvApp theApp;