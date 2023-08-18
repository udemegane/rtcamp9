#pragma once

#include <dxcapi.h>
#include <atlbase.h>
#include <atlcom.h>
#include <filesystem>
#include <string>
#include <unordered_set>
#include "utils/strconv2.h"

class CustomIncludeHandler : public IDxcIncludeHandler
{
public:
    HRESULT STDMETHODCALLTYPE LoadSource(_In_ LPCWSTR pFilename, _COM_Outptr_result_maybenull_ IDxcBlob **ppIncludeSource) override
    {
        if (!pUtils)
        {
            HRESULT hres = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&pUtils));
            if (FAILED(hres))
            {
                throw std::runtime_error("Could not init DXC Utiliy");
            }
        }

        CComPtr<IDxcBlobEncoding> pEncoding;
        std::string path = wide_to_utf8(pFilename);
        if (IncludedFiles.find(path) != IncludedFiles.end())
        {
            // Return empty string blob if this file has been included before
            static const char nullStr[] = " ";
            pUtils->CreateBlobFromPinned(nullStr, ARRAYSIZE(nullStr), DXC_CP_ACP, &pEncoding);
            *ppIncludeSource = pEncoding.Detach();
            return S_OK;
        }

        HRESULT hr = pUtils->LoadFile(pFilename, nullptr, &pEncoding);
        if (SUCCEEDED(hr))
        {
            IncludedFiles.insert(path);
            *ppIncludeSource = pEncoding.Detach();
        }
        return hr;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, _COM_Outptr_ void __RPC_FAR *__RPC_FAR *ppvObject) override { return E_NOINTERFACE; }
    ULONG STDMETHODCALLTYPE AddRef(void) override { return 0; }
    ULONG STDMETHODCALLTYPE Release(void) override { return 0; }
    CComPtr<IDxcUtils> pUtils;
    std::unordered_set<std::string> IncludedFiles;
};