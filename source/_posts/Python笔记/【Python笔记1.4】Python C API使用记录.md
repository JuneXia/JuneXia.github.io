---
title: 【Python笔记1.4】Python C API使用记录
date: 2018-11-02
tags:
categories: ["Python笔记"]
mathjax: true
---

本节知识点：
 1. 用Python/C API实现Python class的实例化，并调用Python class的成员函数;
 2. Python class的\__init__函数需要传参；
 3. Python class的成员函数需要传递列表，解析Python返回的列表。
<!-- more -->

obj_rec.py
```python
class ObjRec:
    def __init__(self, path, threshold=0.5):
        print('do something, ', path, threshold)

    def predict(self, path, param_list):
        print('do something, ', path, param_list)
        return [[0.1, 0.9]]
```


obj_rec.hpp
```c++
class ObjRec
{
	private:
		PyObject *m_pDict = NULL;
		PyObject *m_pHandle = NULL;

	public:
		ObjRec();
		~ObjRec();

		void predict();

};
```

obj_rec.cpp
```c++
void ObjRec::ObjRec()
{
	PyObject* pFile = NULL;
	PyObject* pModule = NULL;
	PyObject* pClass = NULL;
	PyObject* pInitArgs = NULL;

	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
	Py_BEGIN_ALLOW_THREADS;
	Py_BLOCK_THREADS;

	do
	{
#if 0  // 放到主线程中去。
		Py_Initialize();
		if (!Py_IsInitialized())
		{
			printf("Py_Initialize error!\n");
			break;
		}
#endif

		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('/home/user/***')");

		pFile = PyString_FromString("obj_rec");
		pModule = PyImport_Import(pFile);
		if (!pModule)
		{
			LOG_DEBUG("PyImport_Import obj_rec.py failed!\n");
			break;
		}

		m_pDict = PyModule_GetDict(pModule);
		if (!m_pDict)
		{
			LOG_DEBUG("PyModule_GetDict obj_rec.py failed!\n");
			break;
		}

		pClass = PyDict_GetItemString(m_pDict, "ObjRec");
		if (!pClass || !PyCallable_Check(pClass))
		{
			LOG_DEBUG("PyDict_GetItemString ObjRec failed!\n");
			break;
		}

		// PyInstance_New实例化Python类，并在实例化的时候传递参数。
		pInitArgs = PyTuple_New(2);
		if (!pInitArgs)
		{
			LOG_DEBUG("PyTuple_New failed!\n");
			break;
		}
		PyTuple_SetItem(pInitArgs, 0, Py_BuildValue("s", "your string"));
		PyTuple_SetItem(pInitArgs, 1, Py_BuildValue("f", 0.5));
		m_pHandle = PyInstance_New(pClass, pInitArgs, NULL);
		if (!m_pHandle)
		{
			LOG_DEBUG("PyInstance_New failed!\n");
			break;
		}
	} while (0);

	if (pInitArgs)
		Py_DECREF(pInitArgs);
	if (pClass)
		Py_DECREF(pClass);
	if (pModule)
		Py_DECREF(pModule);
	if (pFile)
		Py_DECREF(pFile);

	Py_UNBLOCK_THREADS;
	Py_END_ALLOW_THREADS;
	PyGILState_Release(gstate);
}

ObjRec::~ObjRec()
{
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
	Py_BEGIN_ALLOW_THREADS;
	Py_BLOCK_THREADS;

	if (m_pHandle)
		Py_DECREF(m_pHandle);
	if (m_pDict)
		Py_DECREF(m_pDict);

	Py_UNBLOCK_THREADS;
	Py_END_ALLOW_THREADS;
	PyGILState_Release(gstate);

#if 0  // 放到主线程中去。
	Py_Finalize();
#endif
	LOG_DEBUG("ObjRec::~ObjRec() end!\n");
}

void ObjRec::predict()
{
	PyObject* pArgsDict = NULL;
	PyObject* pArgsList = NULL;

	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
	Py_BEGIN_ALLOW_THREADS;
	Py_BLOCK_THREADS;

	do
	{
		pArgsDict = PyDict_New();
		pArgsList = PyList_New(0);
		if (!pArgsDict || !pArgsList)
		{
			LOG_DEBUG("PyDict_New or PyList_New failed!\n");
			break;
		}

		// PyObject_CallMethod 调用Python类的成员函数，并传递字符串、列表等参数
		PyDict_SetItemString(pArgsDict, "fVal", Py_BuildValue("f", 0.5));
		PyDict_SetItemString(pArgsDict, "flag", Py_BuildValue("b", true));
		PyList_Append(pArgsList, Py_BuildValue("O", pArgsDict));
		PyObject* pRslt = PyObject_CallMethod(m_pHandle, (char *)"predict", (char *)"sO", "your path", pArgsList);

		// 解析Python返回的列表
		int size = PyList_Size(pRslt);
		for (int i=0; i < size; i++)
		{
			float fReading, fConfidence;
			PyObject *pList = NULL, *pVal = NULL;

			pList = PyList_GetItem(pRslt, i);
			pVal = PyList_GetItem(pList, 0);
			PyArg_Parse(pVal, "f", &fReading);
			pVal = PyList_GetItem(pList, 1);
			PyArg_Parse(pVal, "f", &fConfidence);

			std::cout << "i = " << i;
			std::cout << ", fReading = " << fReading;
			std::cout << ", fConfidence = " << fConfidence << std::endl;

			if (pVal)
				Py_DECREF(pVal);
			if (pList)
				Py_DECREF(pList);
		}

		if (pRslt)
			Py_DECREF(pRslt);
	} while(0);

	if (pArgsDict)
		Py_DECREF(pArgsDict);
	if (pArgsList)
		Py_DECREF(pArgsList);

	Py_UNBLOCK_THREADS;
	Py_END_ALLOW_THREADS;
	PyGILState_Release(gstate);
}
```
