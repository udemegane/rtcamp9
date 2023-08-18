#pragma once

#include <slang-com-helper.h>
#include <slang-com-ptr.h>
#include <slang.h>
#include <dxcapi.h>
#include <atlbase.h>
#include <atlcom.h>
#include <tuple>
#include <vector>
#include <string>

#include "utils/strconv2.h"
#include "custom_include_handler.hpp"

class SlangShaderCompiler
{
public:
    SlangShaderCompiler(const char *profile = "glsl_460");
    Slang::ComPtr<slang::IBlob> compile(std::string moduleName, std::string entryName);

private:
    Slang::ComPtr<slang::IGlobalSession> _slangGlobalSession;
    Slang::ComPtr<slang::ISession> _mainSession;
};

class HLSLShaderCompiler
{
public:
    HLSLShaderCompiler(const wchar_t *profile = L"cs_6_6");
    CComPtr<IDxcBlob> compile(std::wstring filename, std::wstring entryName);

private:
    CComPtr<IDxcLibrary> _library;
    CComPtr<IDxcCompiler3> _compiler;
    CComPtr<IDxcUtils> _utils;
    std::unique_ptr<CustomIncludeHandler> _incHandler;
    const wchar_t *_profile;
};