
// HWOpencv.h : PROJECT_NAME ���ε{�����D�n���Y��
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�� PCH �]�t���ɮ׫e���]�t 'stdafx.h'"
#endif

#include "resource.h"		// �D�n�Ÿ�


// CHWOpencvApp: 
// �аѾ\��@�����O�� HWOpencv.cpp
//

class CHWOpencvApp : public CWinApp
{
public:
	CHWOpencvApp();

// �мg
public:
	virtual BOOL InitInstance();

// �{���X��@

	DECLARE_MESSAGE_MAP()
};

extern CHWOpencvApp theApp;