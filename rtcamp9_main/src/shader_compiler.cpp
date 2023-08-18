#pragma once
#include <filesystem>
#include <iostream>
#include "shader_compiler.hpp"

SlangShaderCompiler::SlangShaderCompiler(const char *profile)
{
    SLANG_FAILED(slang::createGlobalSession(_slangGlobalSession.writeRef()));

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = _slangGlobalSession->findProfile(profile);
    targetDesc.forceGLSLScalarBufferLayout = true;
    // spdlog::info("Initialize slang global session...");
    // spdlog::info(" - format: SPIR-V");
    // spdlog::info(" - glsl profile: {}", profile);
    // spdlog::info(" - GLSL Scalar buffer layout: ", targetDesc.forceGLSLScalarBufferLayout ? "enable" : "disable");
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    auto currentDir = std::filesystem::current_path();
    std::vector<std::filesystem::path> shaderDirectories = {
        currentDir,
        currentDir / "shaders",
    };
    std::vector<char *> slangSearchPaths;
    for (auto &path : shaderDirectories)
    {
        auto tmp = path.string();
        slangSearchPaths.push_back(path.string().data());
    }
    const char *searchPaths[] = {"E:/nvpro/bin_x64/Debug/shaders/"};
    sessionDesc.searchPaths = searchPaths; // slangSearchPaths.data();
    sessionDesc.searchPathCount = 1;       //(SlangInt) slangSearchPaths.size();

    SLANG_FAILED(_slangGlobalSession->createSession(sessionDesc, _mainSession.writeRef()));
}

Slang::ComPtr<slang::IBlob> SlangShaderCompiler::compile(std::string moduleName,
                                                         std::string entryName)
{

    slang::IModule *slangModule = nullptr;
    {
        Slang::ComPtr<slang::IBlob> diagnosticBlob;
        slangModule = _mainSession->loadModule(moduleName.c_str(), diagnosticBlob.writeRef());
        if (diagnosticBlob != nullptr)
        {
            fprintf(stderr, "%s\n", (const char *)diagnosticBlob->getBufferPointer());
        }
        if (slangModule == nullptr)
            SLANG_FAILED(slangModule);
        // return std::tuple<const uint32_t *, const size_t>(nullptr, 0);
    }
    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    slangModule->findEntryPointByName(entryName.c_str(), entryPoint.writeRef());

    std::vector<slang::IComponentType *> componentTypes;
    componentTypes.push_back(slangModule);
    componentTypes.push_back(entryPoint);

    Slang::ComPtr<slang::IComponentType> composedProgram;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result =
            _mainSession->createCompositeComponentType(&componentTypes[0], (SlangInt)componentTypes.size(),
                                                       composedProgram.writeRef(), diagnosticsBlob.writeRef());
        SLANG_FAILED(result);
    }

    Slang::ComPtr<slang::IBlob> spirvCode;
    {
        Slang::ComPtr<slang::IBlob> diagnositcsBlob;
        SlangResult result = composedProgram->getEntryPointCode(0, 0, spirvCode.writeRef(), diagnositcsBlob.writeRef());
        SLANG_FAILED(result);
    }
    return spirvCode;
    // return std::tuple<const uint32_t *, const size_t>(static_cast<const uint32_t *>(spirvCode->getBufferPointer()),
    //                                                   spirvCode->getBufferSize());
}

HLSLShaderCompiler::HLSLShaderCompiler(const wchar_t *profile)
{
    _profile = profile;
    HRESULT hres;
    hres = DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&_library));
    if (FAILED(hres))
    {
        throw std::runtime_error("Could not init DXC Library");
    }

    // Initialize DXC compiler
    hres = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&_compiler));
    if (FAILED(hres))
    {
        throw std::runtime_error("Could not init DXC Compiler");
    }

    // Initialize DXC utility
    hres = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&_utils));
    if (FAILED(hres))
    {
        throw std::runtime_error("Could not init DXC Utiliy");
    }
    _incHandler = std::make_unique<CustomIncludeHandler>();
}

CComPtr<IDxcBlob> HLSLShaderCompiler::compile(std::wstring filename, std::wstring entryname)
{
    HRESULT hres;
    auto currentDir = std::filesystem::current_path();
    std::filesystem::path shaderDirectory = currentDir / "shaders";
    auto wdir = utf8_to_wide(shaderDirectory.string());
    filename = wdir + L"/" + filename;

    // Load the HLSL text shader from disk
    uint32_t codePage = DXC_CP_ACP;
    CComPtr<IDxcBlobEncoding> sourceBlob;
    hres = _utils->LoadFile(filename.c_str(), &codePage, &sourceBlob);
    if (FAILED(hres))
    {
        throw std::runtime_error("Could not load shader file");
    }

    // Configure the compiler arguments for compiling the HLSL shader to SPIR-V
    std::vector<LPCWSTR> arguments = {
        // (Optional) name of the shader file to be displayed e.g. in an error message
        filename.c_str(),
        // Shader main entry point
        L"-E", entryname.c_str(),
        // Shader target profile
        L"-T", _profile,
        // Shader include directory
        L"-I", wdir.c_str(),
        // Compile to SPIRV
        L"-spirv"};

    DxcBuffer buffer{};
    buffer.Encoding = DXC_CP_ACP;
    buffer.Ptr = sourceBlob->GetBufferPointer();
    buffer.Size = sourceBlob->GetBufferSize();

    CComPtr<IDxcResult> result{nullptr};
    hres = _compiler->Compile(
        &buffer,
        arguments.data(),
        (uint32_t)arguments.size(),
        _incHandler.get(),
        IID_PPV_ARGS(&result));

    if (SUCCEEDED(hres))
    {
        result->GetStatus(&hres);
    }

    // Output error if compilation failed
    if (FAILED(hres) && (result))
    {
        CComPtr<IDxcBlobEncoding> errorBlob;
        hres = result->GetErrorBuffer(&errorBlob);
        if (SUCCEEDED(hres) && errorBlob)
        {
            std::cerr << "Shader compilation failed :\n\n"
                      << (const char *)errorBlob->GetBufferPointer();
            throw std::runtime_error("Compilation failed");
        }
    }

    // Get compilation result
    CComPtr<IDxcBlob> code;
    result->GetResult(&code);

    return code;
}