import numpy, numba
import os, pathlib, multiprocessing, subprocess, logging, math, uuid, json, time, argparse

#Border Autocrop 2 R8
#Requires pyVips, Numpy, Numba
#Requires external libVips library
#Can resize images using Magic Kernel 2013 / 6 Sharp 7
#Run the script first time to generate JSON with settings
#resizeMode 1 - Numpy MKS 67, 2 - Numpy Lanczos3, 3 - Numpy MKS2013, 4 - Vips Lanczos3

vipsImportSuccess = True

try:
    import pyvips
except:
    vipsImportSuccess = False
    
def avifGetCmdline(inPath: pathlib.Path, outPath: pathlib.Path, argSpeed: int, argQuality: int, argParams: dict):
    avifEncoderPath = pathlib.Path(argParams["avifEncoderPath"])
    tmpList = [str(avifEncoderPath)]
    tmpList.extend(("-s", str(argSpeed), "-q", str(argQuality), "-j", "1", "-d", "12", "-a", "tune=iq"))
    tmpList.append(inPath.as_posix())
    tmpList.append(outPath.as_posix())
    return tmpList

def jxlGetCmdline(inPath: pathlib.Path, outPath: pathlib.Path, argSpeed: int, argQuality: int, argParams: dict):
    jxlEncoderPath = pathlib.Path(argParams["jxlEncoderPath"])
    tmpList = [str(jxlEncoderPath)]
    tmpList.append("\\\\?\\" + str(inPath))
    tmpList.append("\\\\?\\" + str(outPath))
    tmpList.extend(("-e", str(argSpeed), "-d", str(argQuality), "--num_threads=0", "--allow_jpeg_reconstruction", "0"))
    return tmpList

def heifGetCmdline(inPath: pathlib.Path, outPath: pathlib.Path, argSpeed: str, argQuality: int, argParams: dict):
    heifEncoderPath = pathlib.Path(argParams["heifEncoderPath"])
    tmpList = [str(heifEncoderPath)]
    tmpList.extend(("-q", str(argQuality), "--no-alpha", "-b", "12", "-p", "preset=" + argSpeed, "-p", "chroma=444"))
    tmpList.append(inPath.as_posix())
    tmpList.append("-o")
    tmpList.append(outPath.as_posix())
    return tmpList

def checkColor(argImg: pyvips.Image, argImgFormat: str, argParams: dict) -> bool:
    if argImgFormat == "uchar":
        currentThreshold = round(255 * argParams["colorThresholdPercent"] / 100)
        imWkArray = argImg.numpy().astype(numpy.int16)
    else:
        currentThreshold = round(65535 * argParams["colorThresholdPercent"] / 100)
        imWkArray = argImg.numpy().astype(numpy.int32)

    if numpy.max(numpy.absolute(imWkArray[:, :, 0] - imWkArray[:, :, 1])) > currentThreshold or \
            numpy.max(numpy.absolute(imWkArray[:, :, 1] - imWkArray[:, :, 2])) > currentThreshold:
        return True
    return False

def getResultFilePath(argExtension: str, argParams: dict) -> pathlib.Path:
    sourceFilePath = pathlib.Path(argParams["imageFilePath"])
    resultFolderPath = pathlib.Path(argParams["resultFolderPath"])
    return resultFolderPath.joinpath(argParams["resultFolderTemplate"] + argExtension.title(), sourceFilePath.parent.name, \
                sourceFilePath.stem + "." + argExtension.lower())

def resampleImage(argImgArray: numpy.ndarray, oldSize: tuple[int, int], newSize: tuple[int, int], bands, format, kernFunc, supportValue: float, linear: bool = False) -> numpy.ndarray:
    maxColorOut = 65535
    maxColorIn = 255 if format == "uchar" else 65535

    if bands == 1:
        return resampleBand(argImgArray, oldSize, newSize, kernFunc, supportValue, maxColorIn, maxColorOut, linear)
    
    resultArray = numpy.empty((newSize[1], newSize[0], bands), dtype=numpy.uint16)
    for i in range(bands):
        resultArray[:, :, i] = resampleBand(argImgArray[:, :, i], oldSize, newSize, kernFunc, supportValue, maxColorIn, maxColorOut, linear)

    return resultArray

@numba.njit
def resampleBand(argImgArray, oldSize, newSize, kernFunc, supportValue, maxColorIn, maxColorOut, linear) -> numpy.ndarray:
    oldX, oldY = oldSize
    newX, newY = newSize
    imgArrayNormal = numpy.empty((oldY, oldX), dtype=numpy.float64)
    resultArray1Stage = numpy.empty((oldY, newX), dtype=numpy.float64)
    resultArray2Stage = numpy.empty((newY, newX), dtype=numpy.uint16)
    
    if linear:
        for index, i in numpy.ndenumerate(argImgArray):
            imgArrayNormal[index] = sRGBPixelValueToLinear(i, maxColorIn)
    else:
        imgArrayNormal = argImgArray / maxColorIn
    
    weightsMap = getWeightsList(oldX, newX, supportValue, kernFunc)

    for currentYCoord in range(oldY):
        for resultPixelCoord in range(newX):
            internalSum = 0.0
            for currenSourcePixelCoord, currentWeight in weightsMap[resultPixelCoord]:
                internalSum += imgArrayNormal[currentYCoord, currenSourcePixelCoord] * currentWeight
     
            resultArray1Stage[currentYCoord, resultPixelCoord] = internalSum

    weightsMap = getWeightsList(oldY, newY, supportValue, kernFunc)

    for currentXCoord in range(newX):
        for resultPixelCoord in range(newY):
            internalSum = 0.0
            for currenSourcePixelCoord, currentWeight in weightsMap[resultPixelCoord]:
                internalSum += resultArray1Stage[currenSourcePixelCoord, currentXCoord] * currentWeight

            if linear:
                internalSumFinal = linearPixelValueToSRGBNormal(internalSum)
            else:
                internalSumFinal = internalSum

            resultArray2Stage[resultPixelCoord, currentXCoord] = clampZero(round(internalSumFinal * maxColorOut), maxColorOut) 

    return resultArray2Stage

@numba.njit
def getWeightsList(oldSize: int, newSize: int, supportValue: float, kernFunc) -> list[(int, float)]:
    weightsMap = []
    scaleFactor = newSize / oldSize
    scaleFactorClamped = min(1.0, scaleFactor)
    srcWindow = supportValue / scaleFactorClamped

    for resultPixelCoord in range(newSize):
        sourceCenterPixel = (resultPixelCoord + 0.5) / scaleFactor
        firstSourcePixelCoord = math.floor(sourceCenterPixel - srcWindow)
        lastSourcePixelCoord = math.ceil(sourceCenterPixel + srcWindow)
        
        weights = [kernFunc((sourcePixelCoord + 0.5 - sourceCenterPixel) * scaleFactorClamped) \
                    for sourcePixelCoord in range(firstSourcePixelCoord, lastSourcePixelCoord + 1)]
        internalSum = sum(weights)
        weightsMap.append([(clampZero(index + firstSourcePixelCoord, oldSize - 1), cWeight / internalSum) \
                            for index, cWeight in enumerate(weights) if cWeight != 0.0])

    return weightsMap

@numba.njit
def clampZero(inputInt: int, maxInt: int):
    if inputInt < 0:
        return 0
    elif inputInt > maxInt:
        return maxInt
    return inputInt

@numba.njit
def sRGBPixelValueToLinear(s_in, maxColors):
    s = s_in / maxColors
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4

@numba.njit
def sRGBPixelValueToLinearNormal(s):
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4

@numba.njit
def linearPixelValueToSRGB(s, maxColors):
    if s <= 0.0031308:
        return (12.92 * s) * maxColors
    return (1.055 * s ** (1 / 2.4) - 0.055) * maxColors
   
@numba.njit
def linearPixelValueToSRGBNormal(s):
    if s <= 0.0031308:
        return 12.92 * s
    return (1.055 * s ** (1 / 2.4) - 0.055)

@numba.njit
def lanc3Value(x: float) -> float:
    if x > -3 and x < 3:
        return numpy.sinc(x) * numpy.sinc(x / 3)
    return 0

@numba.njit
def MKS2013Value(x: float) -> float:
    if x < 0.0:
        x = -x
    if  x <= 0.5:
        return 17.0 / 16.0 - (7.0 / 4.0) * x ** 2
    elif x <= 1.5:
        return 0.25 * (4.0 * x ** 2 - 11.0 * x + 7.0)
    elif x <= 2.5:
        return -0.125 * (x - 2.5) * (x - 2.5)
    return 0.0

@numba.njit
def MKS67Value(x: float) -> float:
    if x < -10 or x >= 10:
        return 0

    internalSum = 0.0
    for s in range(-7, 8):
        internalSum += mkSharp67Helper(s) * mk67Helper(x + s)
    return internalSum

@numba.njit
def mkSharp67Helper(offset) -> float:
    match abs(offset):
        case 0:
            return 8415120629 / 2952708336
        case 1:
            return -3913360945 / 2952708336
        case 2:
            return 20369223277 / 35432500032
        case 3:
            return -4387620835 / 17716250016
        case 4:
            return 1887357215 / 17716250016
        case 5:
            return -808049411 / 17716250016
        case 6:
            return 674477135 / 35432500032
        case 7:
            return -3769012 / 553632813
        case _:
            return 0

@numba.njit
def mk67Helper(x: float) -> float:
    if x < -3 or x >= 3:
        return 0
    
    if x <= 0:
        if x <= -2:
            return x**5 / 120 + x**4 / 8 + 3 * x**3 / 4 + 9 * x**2 / 4 + 27 * x / 8 + 81 / 40
        if x <= -1:
            return -(x**5) / 24 - 3 * x**4 / 8 - 5 * x**3 / 4 - 7 * x**2 / 4 - 5 * x / 8 + 17 / 40
        return x**5 / 12 + x**4 / 4 - x**2 / 2 + 11 / 20
    
    if x <= 1:
        return -(x**5) / 12 + x**4 / 4 - x**2 / 2 + 11 / 20
    
    if x <= 2:
        return x**5 / 24 - 3 * x**4 / 8 + 5 * x**3 / 4 - 7 * x**2 / 4 + 5 * x / 8 + 17 / 40
    
    return -(x**5) / 120 + x**4 / 8 - 3 * x**3 / 4 + 9 * x**2 / 4 - 27 * x / 8 + 81 / 40

@numba.njit
def getResampleSize(imgSize: tuple[int, int], argVTarget, argHTarget) -> tuple[int, int]:
    x, y = imgSize
    internalTarget = argVTarget if x < y else argHTarget

    if x > internalTarget:
        tmpRatio = internalTarget / x
        newY = round(y * tmpRatio)
        return internalTarget, newY
    return x, y

def getColorBounds(argInputColor: int, argDistance: int) -> tuple[int, int]:
    leftBias, rightBias = 0, 0
    singleSideDistance = round(argDistance / 2)
    if argInputColor - singleSideDistance < 0:
        rightBias = abs(argInputColor - singleSideDistance)
    elif argInputColor + singleSideDistance > 255:
        leftBias = abs(argInputColor + singleSideDistance - 255)
    leftBound = max(0, argInputColor - singleSideDistance - leftBias) 
    rightBound = min(255, argInputColor + singleSideDistance + rightBias)
    return leftBound, rightBound

def cropUnified(argImageArray: numpy.ndarray, argVertical: bool, argExhaustive: bool, argReverse: bool, argColor: bool, argParams: dict):
    emptyLinesList = []

    if argColor:
        fuzzyDistance = argParams["colorFuzzyDistance"]
        lineErrorThreshold = argParams["colorLineThreshold"]
    else:
        if argExhaustive:
            fuzzyDistance = argParams["monoFuzzyDistanceEx"]
        else:
            fuzzyDistance = argParams["monoFuzzyDistance"]
        lineErrorThreshold = argParams["monoLineThreshold"]        

    internalAxis1, internalAxis2 = int(not argVertical), int(argVertical)    

    if argExhaustive:
        internalRange = range(0, numpy.size(argImageArray, internalAxis1))
        internalSize1 = numpy.size(argImageArray, internalAxis1)
    else:
        if argReverse:
            internalRange = range(numpy.size(argImageArray, internalAxis1) - 1, -1, -1)
        else:
            internalRange = range(0, numpy.size(argImageArray, internalAxis1))
    internalSize2 = numpy.size(argImageArray, internalAxis2)   

    for coord in internalRange:
        if argVertical:
            vals, freq = numpy.unique(argImageArray[coord, :], return_counts=True)
        else:
            vals, freq = numpy.unique(argImageArray[:, coord], return_counts=True)
        backColor = numpy.int64(vals[numpy.argmax(freq)])
        leftBound, rightBound = getColorBounds(backColor, fuzzyDistance)
        newFuzzyCount = numpy.sum(numpy.extract(numpy.logical_and(vals >= leftBound, vals <= rightBound), freq))
        lineFuzzyError = (1 - newFuzzyCount / internalSize2) * 100
        if lineFuzzyError < lineErrorThreshold:
            emptyLinesList.append(coord)
        else:
            if not argExhaustive:
                return emptyLinesList, lineFuzzyError
            
    if argExhaustive:
        if len(emptyLinesList) == internalSize1:
            emptyLinesList.clear()
    else:
        emptyLinesList.clear()

    return emptyLinesList, -1

def workerEntryPoint(argParams: dict):
    tmrLocal = time.time()
    errorMessagesList = []
    debugMessagesList = []
    argParams["debugList"] = debugMessagesList
    argParams["errorList"] = errorMessagesList

    im = pyvips.Image.new_from_file(argParams["imageFilePath"], memory=True, access="random")
    imWidth = im.get("width")
    imHeight = im.get("height")
    imBands = im.get("bands")
    imPixelSize = im.get("format")
    imColorFormat = im.get("interpretation")
    debugMessagesList.append(f"Loaded {argParams["imageFilePath"]}")
    debugMessagesList.append(f"Size: {imWidth}x{imHeight}, Bands: {imBands}, Pixel size: {imPixelSize}, Image type: {imColorFormat}")

    if im.hasalpha(): #yank alpha with all the force needed
        im = im[:-1]
        imBands -= 1
        debugMessagesList.append("Alpha channel removed.")

    if imBands == 1:
        imColor = False
    else:
        imColor = checkColor(im, imPixelSize, argParams)
        if not imColor:
            imBands = 1
            debugMessagesList.append("Grayscale-in-RGB image detected, converting to grayscale.")
            if imPixelSize == "uchar":
                im = im.colourspace(pyvips.Interpretation.B_W)
            else:
                im = im.colourspace(pyvips.Interpretation.GREY16)
            imColorFormat = im.get("interpretation")

    imArray = im.numpy()

    if argParams["doCrop"]:
        if imColorFormat == pyvips.Interpretation.B_W:
            imWkArray = numpy.copy(imArray)
        else:
            imWkArray = im.colourspace(pyvips.Interpretation.B_W).numpy()

        topLinesList, topFuzzyError = cropUnified(imWkArray, True, False, False, imColor, argParams)
        bottomLinesList, bottomFuzzyError = cropUnified(imWkArray, True, False, True, imColor, argParams)

        if len(topLinesList) == 0:
            if topFuzzyError == -1:
                debugMessagesList.append("Fast crop (top): probably empty image")
            else:
                debugMessagesList.append(f"Fast crop (top): nothing cropped, Fe: {topFuzzyError:.4f}%")
        else:
            debugMessagesList.append(f"Fast crop (top): content line {topLinesList[-1]}, Fe: {topFuzzyError:.4f}%")
        if len(bottomLinesList) == 0:
            if bottomFuzzyError == -1:
                debugMessagesList.append("Fast crop (bottom): probably empty image")
            else:
                debugMessagesList.append(f"Fast crop (bottom): nothing cropped, Fe: {bottomFuzzyError:.4f}%")  
        else:
            debugMessagesList.append(f"Fast crop (bottom): content line {bottomLinesList[-1]}, {len(bottomLinesList)} lines, Fe: {bottomFuzzyError:.4f}%")
        imArray = numpy.delete(imArray, topLinesList + bottomLinesList, 0)

        if argParams["excludeVerticalCrop"]:
            imWkArray = numpy.delete(imWkArray, topLinesList + bottomLinesList, 0)

        leftLinesList, leftFuzzyError = cropUnified(imWkArray, False, False, False, imColor, argParams)
        rightLinesList, rightFuzzyError = cropUnified(imWkArray, False, False, True, imColor, argParams)

        if (((len(leftLinesList) + len(rightLinesList)) / imWidth) < (argParams["exhaustiveThreshold"] / 100)) and argParams["enableExhaustive"]:
            lineListH, _ = cropUnified(imWkArray, False, True, False, imColor, argParams)
            debugMessagesList.append(f"Exhaustive crop (horizontal): {len(lineListH)} lines matched")
            imArray = numpy.delete(imArray, lineListH, 1)
        else:
            if len(leftLinesList) == 0:
                if leftFuzzyError == -1:
                    debugMessagesList.append("Fast crop (left): probably empty image")
                else:
                    debugMessagesList.append(f"Fast crop (left): nothing cropped, Fe: {leftFuzzyError:.4f}%")  
            else:
                debugMessagesList.append(f"Fast crop (left): content line {leftLinesList[-1]}, Fe: {leftFuzzyError:.4f}%")
            if len(rightLinesList) == 0:
                if leftFuzzyError == -1:
                    debugMessagesList.append("Fast crop (right): probably empty image")
                else:
                    debugMessagesList.append(f"Fast crop (right): nothing cropped, Fe: {rightFuzzyError:.4f}%")  
            else:
                debugMessagesList.append(f"Fast crop (right): content line {rightLinesList[-1]}, {len(rightLinesList)} lines, Fe: {rightFuzzyError:.4f}%")
            imArray = numpy.delete(imArray, leftLinesList + rightLinesList, 1)

    if argParams["doResize"]:
        oldSize = (imArray.shape[1], imArray.shape[0])
        newSize = getResampleSize(oldSize, argParams["verticalResizeTarget"], argParams["horizontalResizeTarget"])
        if oldSize == newSize:
            imResizeImg = pyvips.Image.new_from_array(imArray, interpretation="auto")
            debugMessagesList.append(f"Image is too small to resize ({oldSize[0]}x{oldSize[1]})")
        else:
            resizeMode = argParams["resizeMode"]
            if resizeMode <= 3:
                if resizeMode == 1:
                    currentKernel = MKS67Value
                    currentSpValue = 10
                elif resizeMode == 2:
                    currentKernel = lanc3Value
                    currentSpValue = 3
                elif resizeMode == 3:
                    currentKernel = MKS2013Value
                    currentSpValue = 3
                imResizeImg = pyvips.Image.new_from_array( \
                    resampleImage(imArray, oldSize, newSize, imBands, imPixelSize, currentKernel, currentSpValue, argParams["gammaCompensation"]), interpretation="auto")
            else:
                resizeRatio = newSize[0] / oldSize[0]
                match imColorFormat:
                    case "b-w":
                        imResizeImg = pyvips.Image.new_from_array(imArray, interpretation="auto").colourspace("grey16") \
                            .resize(resizeRatio, kernel=pyvips.Kernel.LANCZOS3)
                    case "srgb":
                        imResizeImg = pyvips.Image.new_from_array(imArray, interpretation="auto").colourspace("rgb16") \
                            .resize(resizeRatio, kernel=pyvips.Kernel.LANCZOS3)
                    case _:
                        imResizeImg = pyvips.Image.new_from_array(imArray, interpretation="auto").resize(resizeRatio, kernel=pyvips.Kernel.LANCZOS3)
            debugMessagesList.append(f"Resizing from {oldSize[0]}x{oldSize[1]} to {newSize[0]}x{newSize[1]}")
    else:
        imResizeImg = pyvips.Image.new_from_array(imArray, interpretation="auto")
    
    if argParams["doEncodePng"]:
        exportPng(imResizeImg, argParams)
    if argParams["doEncodeJxl"]:
        exportUnifiedSecant(imResizeImg, "jxl", argParams)
    if argParams["doEncodeAvif"]:
        exportUnified(imResizeImg, "avif", argParams)
    if argParams["doEncodeHeif"]:
        exportUnified(imResizeImg, "heif", argParams)

    debugMessagesList.append(f"Local runtime: {time.time() - tmrLocal}")

    return argParams["imageFilePath"], debugMessagesList, errorMessagesList

def exportPng(argImg: pyvips.Image, argParams: dict):
    debugList = argParams["debugList"]
    resultFilePath = getResultFilePath("png", argParams)
    os.makedirs(resultFilePath.parent, exist_ok=True)
    argImg.write_to_file(resultFilePath, compression=argParams["pngCompressionLevel"])
    debugList.append("Encoded to PNG.")

def exportUnified(argImg: pyvips.Image, argCodecId: str, argParams: dict):
    debugList = argParams["debugList"]

    resultFilePath = getResultFilePath(argCodecId, argParams)
    os.makedirs(resultFilePath.parent, exist_ok=True)
    tmpPngFile = resultFilePath.with_name(str(uuid.uuid4()) + ".png")
    argImg.write_to_file(tmpPngFile, compression=1)

    match argCodecId:
        case "avif":
            minQuality = 0
            maxQuality = 62
            maxBound = 63
        case "jxl":
            minQuality = 25.0
            maxQuality = 0.0
        case "heif":
            minQuality = 0
            maxQuality = 51
            maxBound = 51    

    forceStatsFlag = argParams[argCodecId + "ForceStatistics"]
    enableApproxFlag = argParams[argCodecId + "EnableApprox"]
    if forceStatsFlag or enableApproxFlag:
        ssimDiffList = argParams[argCodecId + "SsimDiffList"]
        encodeSpeedApprox = argParams[argCodecId + "ApproxSpeed"]
    decoderPath = argParams[argCodecId + "DecoderPath"]
    encodeSpeedNormal = argParams[argCodecId + "EncodeSpeed"]
    if argCodecId == "jxl":
        currentQuality = argParams[argCodecId + "TargetQuality"]
    else:
        currentQuality = fromQtoQuant(argParams[argCodecId + "TargetQuality"], maxBound)    
    encoderCmdlineFunc = globals()[argCodecId + "GetCmdline"]

    if not enableApproxFlag:
        localDecoderPath = decoderPath if forceStatsFlag else None
        localQuality = argParams[argCodecId + "TargetQuality"]
        savedSsim = saveImgUnified(tmpPngFile, resultFilePath, encodeSpeedNormal, localQuality, encoderCmdlineFunc, localDecoderPath, argParams)
        tmpPngFile.unlink()

        if not savedSsim == 101:
            debugList.append("Encoded to " + argCodecId)
            if forceStatsFlag:
                ssimDiffList.append(savedSsim)
        return
    
    biasFlag = argParams[argCodecId + "EnableBias"]
    if encodeSpeedApprox == encodeSpeedNormal and biasFlag:
        biasFlag = False
        debugList.append("Approx speed = Encode speed, disabling bias.")
    if biasFlag:
        biasList = argParams[argCodecId + "BiasList"]

    centerPointList = argParams[argCodecId + "CenterPointList"]
    initialSsim = argParams["targetSsim"]
    iterCountList = argParams[argCodecId + "IterCountList"]
    ssimToleranceInitial = argParams["ssimToleranceInitial"]
    ssimTolerance = argParams["ssimTolerance"]
    jumpDict = argParams[argCodecId + "JumpDict"]
    
    if len(centerPointList) != 0:
        if argCodecId == "jxl":
            currentQuality = numpy.median(centerPointList)
        else:
            currentQuality = round(numpy.median(centerPointList))

    initialQuality = currentQuality
    initialQualityMin = minQuality
    localQuality = currentQuality if argCodecId == "jxl" else fromQuantToQ(currentQuality, maxBound)

    candidateList = []
    previousQuality = 0
    iterCount = 0
    approxSsim = 0
    ssimBias = 0
    iterDone = False

    debugList.append(f"Initial center point: {localQuality}/{currentQuality}")

    if biasFlag:
        if len(biasList) != 0:
            ssimBias = numpy.median(biasList)

    targetSsim = initialSsim - ssimBias
    if biasFlag:
        debugList.append(f"Biased SSIM: {targetSsim}")
    else:
        debugList.append(f"Non-biased SSIM: {targetSsim}")

    while not iterDone:
        dupFound = False
        for _, candQuality, candSsim in candidateList:
            if candQuality == currentQuality:
                approxSsim = candSsim
                dupFound = True
                debugList.append(f"Iter S, Q {localQuality}/{currentQuality}, Ssim {approxSsim}")
            
        if not dupFound:
            iterCount += 1
            candFilePath = resultFilePath.with_name(str(uuid.uuid4()) + "." + argCodecId)
            approxSsim = saveImgUnified(tmpPngFile, candFilePath, encodeSpeedApprox, localQuality, encoderCmdlineFunc, decoderPath, argParams)
            if approxSsim == 101:
                tmpPngFile.unlink(True)
                return
            if iterCount == 1:
                firstApproxSsim = approxSsim
            candidateList.append((candFilePath, currentQuality, approxSsim))
            debugList.append(f"Iter {iterCount}, Q {localQuality}/{currentQuality}, Ssim {approxSsim}")

        currentSsimTolerance = ssimToleranceInitial if iterCount == 1 else ssimTolerance

        if abs(approxSsim - targetSsim) < currentSsimTolerance:
            debugList.append(f"Exit reason: target Ssim < {currentSsimTolerance}")
            iterDone = True

        if (len(candidateList) > 1) and (not iterDone) and (argCodecId != "jxl"):
            candidateList.sort(key=lambda x: x[1])
            for i in range(len(candidateList) - 1):
                prevCand = candidateList[i]
                curCand = candidateList[i + 1]
                if curCand[1] - prevCand[1] == 1:
                    if (prevCand[2] < targetSsim) and (curCand[2] > targetSsim):
                        debugList.append(f"Exit reason: large SSIM difference (Q {prevCand[1]}/{curCand[1]}, Ssim {prevCand[2]}/{curCand[2]})")
                        iterDone = True

        if (abs(currentQuality - initialQualityMin) <= 1) and (approxSsim > targetSsim) and (not iterDone):
            debugList.append("Exit reason: very low quality and very high SSIM score. Probably empty image.")
            iterDone = True
        
        if argCodecId == "jxl" and iterCount > 10:
            debugList.append("Exit reason: iteration count limit reached.")
            iterDone = True        
        
        if not iterDone:
            previousQuality = currentQuality

            if approxSsim > targetSsim:
                maxQuality = currentQuality
            else:
                minQuality = currentQuality

            roundFirstSsim = str(round(approxSsim))
            if (roundFirstSsim in jumpDict) and (not dupFound):
                jumpList = jumpDict[roundFirstSsim]
                if argCodecId == "jxl":
                    jumpValue = numpy.median(jumpList)
                    if approxSsim > targetSsim:
                        currentQuality = min(minQuality, currentQuality + jumpValue)
                    else:
                        currentQuality = max(maxQuality, currentQuality - jumpValue)                        
                else:
                    jumpValue = round(numpy.median(jumpList))
                    if approxSsim > targetSsim:
                        currentQuality = max(minQuality, currentQuality - jumpValue)
                    else:
                        currentQuality = min(maxQuality, currentQuality + jumpValue)
                debugList.append(f"Jump list found for SSIM {roundFirstSsim}, length {len(jumpList)}, value {jumpValue}")
            else:
                if argCodecId == "jxl":
                    currentQuality = (maxQuality + minQuality) / 2
                else:
                    currentQuality = round((maxQuality + minQuality) / 2)

            if currentQuality == previousQuality:               
                debugList.append("Exit reason: current Q = previous Q")
                iterDone = True
            
            localQuality = currentQuality if argCodecId == "jxl" else fromQuantToQ(currentQuality, maxBound)
    
    selectedCandidate = min(candidateList, key=lambda x: abs(x[2] - targetSsim))

    currentQuality = selectedCandidate[1]
    localQuality = currentQuality if argCodecId == "jxl" else fromQuantToQ(currentQuality, maxBound)

    debugList.append(f"Selected Quality {localQuality}/{currentQuality}, Ssim {selectedCandidate[2]}")

    if encodeSpeedApprox == encodeSpeedNormal:
        savedSsim = selectedCandidate[2]
        selectedCandidate[0].replace(resultFilePath)
    else:
        savedSsim = saveImgUnified(tmpPngFile, resultFilePath, encodeSpeedNormal, localQuality, encoderCmdlineFunc, decoderPath, argParams)
        if savedSsim == 101:
            tmpPngFile.unlink(True)
            return

    for candPath, _, _ in candidateList:
        candPath.unlink(True)
    tmpPngFile.unlink()

    ssimDiff = initialSsim - savedSsim

    if biasFlag:
        if ssimBias != 0:
            ssimDiffList.append(savedSsim)
        else:
            debugList.append("Ssim bias = 0, skip adding to statistics.")
        newBias = ssimBias - ssimDiff
        biasList.append(newBias)
        debugList.append(f"New calculated bias: {newBias}")
    else:
        ssimDiffList.append(savedSsim)
    
    iterCountList.append(iterCount)
    centerPointList.append(currentQuality)

    if len(centerPointList) > 100:
        del centerPointList[:len(centerPointList) - 100]

    if iterCount != 1:
        roundFirstSsim = str(round(firstApproxSsim))
        jumpValue = abs(initialQuality - selectedCandidate[1])
        if roundFirstSsim in jumpDict:
            jumpList = jumpDict[roundFirstSsim]
            if len(jumpList) > 100:
                del jumpList[:len(jumpList) - 100]
        else:
            jumpList = []
        if jumpValue == 0:
            jumpValue = 1
        jumpList.append(jumpValue)
        jumpDict[roundFirstSsim] = jumpList
        debugList.append(f"Adding jump value {jumpValue} for SSIM {roundFirstSsim}")
    
    debugList.append(f"Approx SSIM: {approxSsim}, Saved SSIM: {savedSsim}, SSIM Diff: {ssimDiff:.8f}, Quality: {localQuality}/{currentQuality}, Iterations: {iterCount}")
    debugList.append("Encoded to " + argCodecId)    

def fromQuantToQ(quantizer: int, maxBound: int) -> int:
    return round((quantizer / maxBound) * 100)

def fromQtoQuant(quality: int, maxBound: int) -> int:
    return round((quality / 100) * maxBound)

def exportUnifiedSecant(argImg: pyvips.Image, argCodecId: str, argParams: dict):
    debugList = argParams["debugList"]

    resultFilePath = getResultFilePath(argCodecId, argParams)
    os.makedirs(resultFilePath.parent, exist_ok=True)
    tmpPngFile = resultFilePath.with_name(str(uuid.uuid4()) + ".png")
    argImg.write_to_file(tmpPngFile, compression=1)

    match argCodecId:
        case "avif":
            minQuality = 0
            maxQuality = 62
            maxBound = 63
        case "jxl":
            minQuality = 25.0
            maxQuality = 0.0
        case "heif":
            minQuality = 0
            maxQuality = 51
            maxBound = 51

    forceStatsFlag = argParams[argCodecId + "ForceStatistics"]
    enableApproxFlag = argParams[argCodecId + "EnableApprox"]
    if forceStatsFlag or enableApproxFlag:
        ssimDiffList = argParams[argCodecId + "SsimDiffList"]
        encodeSpeedApprox = argParams[argCodecId + "ApproxSpeed"]
    decoderPath = argParams[argCodecId + "DecoderPath"]
    encodeSpeedNormal = argParams[argCodecId + "EncodeSpeed"]
    if argCodecId == "jxl":
        currentQuality = argParams[argCodecId + "TargetQuality"]
    else:
        currentQuality = fromQtoQuant(argParams[argCodecId + "TargetQuality"], maxBound)
    encoderCmdlineFunc = globals()[argCodecId + "GetCmdline"]

    if not enableApproxFlag:
        localDecoderPath = decoderPath if forceStatsFlag else None
        localQuality = argParams[argCodecId + "TargetQuality"]

        savedSsim = saveImgUnified(tmpPngFile, resultFilePath, encodeSpeedNormal, localQuality, encoderCmdlineFunc, localDecoderPath, argParams)
        tmpPngFile.unlink()

        if not savedSsim == 101:
            debugList.append("Encoded to " + argCodecId)
            if forceStatsFlag:
                ssimDiffList.append(savedSsim)
        return
    
    biasFlag = argParams[argCodecId + "EnableBias"]
    if encodeSpeedApprox == encodeSpeedNormal and biasFlag:
        biasFlag = False
        debugList.append("Approx speed = Encode speed, force disabling bias.")
    if biasFlag:
        biasList = argParams[argCodecId + "BiasList"]
    centerPointList = argParams[argCodecId + "CenterPointList"]
    initialSsim = argParams["targetSsim"]
    iterCountList = argParams[argCodecId + "IterCountList"]
    ssimTolerance = argParams["ssimTolerance"]
    jumpDict = argParams[argCodecId + "JumpDict"]

    if len(centerPointList) == 0:
        qInit1 = currentQuality
        if argCodecId == "jxl":
            qInit2 = qInit1 / 2
        else:
            qInit2 = round(qInit1 / 2)
    else:
        if argCodecId == "jxl":
            qInit1 = numpy.median(centerPointList)
            qInit2 = qInit1 / 2
        else:            
            qInit1 = round(numpy.median(centerPointList))
            qInit2 = round(qInit1 / 2)

    iterDone = False
    iterCount = skipCount = 0
    approxSsimDiff = 0
    qIter1 = qIter2 = ssimIter1 = ssimIter2 = 0
    initialQuality = qInit1
    ssimBias = 0
    candidateList = []
    
    if biasFlag:
        if len(biasList) != 0:
            ssimBias = numpy.median(biasList)

    targetSsim = initialSsim - ssimBias
    if biasFlag:
        debugList.append(f"Biased SSIM: {targetSsim}")
    else:
        debugList.append(f"Non-biased SSIM: {targetSsim}")

    if argCodecId == "jxl":
        debugList.append(f"Initial center point: {qInit1}, {qInit2}")
    else:
        debugList.append(f"Initial center point: {fromQuantToQ(qInit1, maxBound)}, {fromQuantToQ(qInit2, maxBound)}")

    while not iterDone:
        match iterCount:
            case 0:
                currentQuality = qInit1
            case 1: 
                roundFirstSsim = str(round(approxSsim))
                if roundFirstSsim in jumpDict:
                    jumpList = jumpDict[roundFirstSsim]
                    if argCodecId == "jxl":
                        jumpValue = numpy.median(jumpList)
                        if approxSsim > targetSsim:
                            currentQuality = min(minQuality, currentQuality + jumpValue)
                        else:
                            currentQuality = max(maxQuality, currentQuality - jumpValue)                        
                    else:
                        jumpValue = round(numpy.median(jumpList))
                        if approxSsim > targetSsim:
                            currentQuality = max(minQuality, currentQuality - jumpValue)
                        else:
                            currentQuality = min(maxQuality, currentQuality + jumpValue)
                    debugList.append(f"Jump list found for SSIM {roundFirstSsim}, length {len(jumpList)}, value {jumpValue}")
                else:
                    currentQuality = qInit2
            case _:
                if argCodecId == "jxl":
                    currentQuality = clampZero(qIter2 - ((qIter2 - qIter1) / (ssimIter2 - ssimIter1)) * ssimIter2, minQuality)
                else:
                    currentQuality = clampZero(round(qIter2 - ((qIter2 - qIter1) / (ssimIter2 - ssimIter1)) * ssimIter2), maxQuality)
        localQuality = currentQuality if argCodecId == "jxl" else fromQuantToQ(currentQuality, maxBound)

        dupFound = False

        for _, curQuality, curSsim in candidateList:
            if curQuality == currentQuality:
                skipCount += 1
                approxSsim = curSsim
                approxSsimDiff = curSsim - targetSsim
                dupFound = True
                debugList.append(f"Iter (Skipped), Q {localQuality}, Ssim {approxSsim}, Ssim diff {approxSsimDiff}")
        
        if not dupFound:
            iterCount += 1

            candFilePath = resultFilePath.with_name(str(uuid.uuid4()) + "." + argCodecId)
            approxSsim = saveImgUnified(tmpPngFile, candFilePath, encodeSpeedApprox, localQuality, encoderCmdlineFunc, decoderPath, argParams)
            approxSsimDiff = approxSsim - targetSsim
            if approxSsim == 101:
                tmpPngFile.unlink(True)
                return
            if iterCount == 1:
                firstApproxSsim = approxSsim
            candidateList.append((candFilePath, currentQuality, approxSsim))

            debugList.append(f"Iter {iterCount}, Q {localQuality}, Ssim {approxSsim}, Ssim diff {approxSsimDiff}")
        
        if abs(approxSsimDiff) < ssimTolerance:
            iterDone = True
            debugList.append(f"Exit reason: approx ssim < {ssimTolerance}")

        if (currentQuality == qIter2) and (not iterDone):
            iterDone = True
            debugList.append(f"Exit reason: current Q = previous Q")

        if (approxSsimDiff == ssimIter2) and (not iterDone):
            iterDone = True
            debugList.append(f"Exit reason: current SSIM = previous SSIM")

        if (skipCount == 5) and (not iterDone):
            iterDone = True
            debugList.append(f"Exit reason: skip count exceeded limit")

        if (iterCount > 10) and (not iterDone):
            iterDone = True
            debugList.append("Exit reason: iteration count exceeded limit")

        if (abs(currentQuality - minQuality) <= 1) and (approxSsim > targetSsim) and (not iterDone):
            iterDone = True
            debugList.append("Exit reason: very low quality and very high SSIM score. Probably empty image.")

        if (len(candidateList) > 1) and (not iterDone) and (argCodecId != "jxl"):
            candidateList.sort(key=lambda x: x[1])
            for i in range(len(candidateList) - 1):
                prevCand = candidateList[i]
                curCand = candidateList[i + 1]
                if curCand[1] - prevCand[1] == 1:
                    if (prevCand[2] < targetSsim) and (curCand[2] > targetSsim):
                        iterDone = True
                        debugList.append(f"Exit reason: Small Q diff, big SSIM diff (Q {prevCand[1]}/{curCand[1]}, Ssim {prevCand[2]}/{curCand[2]})")
                              
        if not iterDone:
            qIter1, qIter2 = qIter2, currentQuality
            ssimIter1, ssimIter2 = ssimIter2, approxSsimDiff

    selectedCandidate = min(candidateList, key=lambda x: abs(x[2] - targetSsim))

    currentQuality = selectedCandidate[1]
    localQuality = currentQuality if argCodecId == "jxl" else fromQuantToQ(currentQuality, maxBound)

    debugList.append(f"Selected Quality {localQuality}, SSIM {selectedCandidate[2]}")    
    
    if encodeSpeedApprox == encodeSpeedNormal:
        savedSsim = selectedCandidate[2]
        selectedCandidate[0].replace(resultFilePath)
    else:
        savedSsim = saveImgUnified(tmpPngFile, resultFilePath, encodeSpeedNormal, localQuality, encoderCmdlineFunc, decoderPath, argParams)
        if savedSsim == 101:
            tmpPngFile.unlink(True)
            return

    for candPath, _, _ in candidateList:
        candPath.unlink(True)
    tmpPngFile.unlink()

    ssimDiff = initialSsim - savedSsim

    if biasFlag:
        if ssimBias != 0:
            ssimDiffList.append(savedSsim)
        else:
            debugList.append("Ssim bias = 0, skip adding to statistics.")
        newBias = ssimBias - ssimDiff
        biasList.append(newBias)
        debugList.append(f"New calculated bias: {newBias}")
    else:
        ssimDiffList.append(savedSsim)

    centerPointList.append(currentQuality)
    iterCountList.append(iterCount)

    if len(centerPointList) > 100:
        del centerPointList[:len(centerPointList) - 100]

    if iterCount != 1:
        roundFirstSsim = str(round(firstApproxSsim))
        jumpValue = abs(initialQuality - selectedCandidate[1])
        if roundFirstSsim in jumpDict:
            jumpList = jumpDict[roundFirstSsim]
            if len(jumpList) > 100:
                del jumpList[:len(jumpList) - 100]
        else:
            jumpList = []
        if jumpValue == 0:
            jumpValue = 1
        jumpList.append(jumpValue)
        jumpDict[roundFirstSsim] = jumpList
        debugList.append(f"Adding jump value {jumpValue} for SSIM {roundFirstSsim}")
    
    debugList.append(f"Approx SSIM: {approxSsim}, Saved SSIM: {savedSsim}, SSIM Diff: {ssimDiff}, Quality: {localQuality}, Iterations: {iterCount}")
    debugList.append("Encoded to " + argCodecId)

def saveImgUnified(argSourceImgPath: pathlib.Path, argResultImgPath: pathlib.Path, argEncSpeed, argEncQuality: int, argEncoderCmdlineFunc, argDecoderPath: pathlib.Path, argParams: dict):
    errorList = argParams["errorList"]
    decodedPngImage = argSourceImgPath.parent.joinpath(str(uuid.uuid4()) + ".png")
    encoderProcess = subprocess.run(argEncoderCmdlineFunc(argSourceImgPath, argResultImgPath, argEncSpeed, argEncQuality, argParams), \
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS, text=True)
    if encoderProcess.returncode != 0:
        errorList.append(f"Encoder error: {encoderProcess.stderr}")
        return 101
    if argDecoderPath == None:
        return 0
    decoderCmdline = [argDecoderPath, argResultImgPath, decodedPngImage]
    decoderProcess = subprocess.run(decoderCmdline, \
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS, text=True)
    if decoderProcess.returncode != 0:
        errorList.append(f"Decoder error: {decoderProcess.stderr}")
        argResultImgPath.unlink(True)
        return 101
    ssimArgList = [argParams["ssimulacraPath"], argSourceImgPath, decodedPngImage]
    ssimulacraProcess = subprocess.run(ssimArgList, \
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS, text=True)
    if ssimulacraProcess.returncode != 0:
        errorList.append(f"Ssimulacra error: {ssimulacraProcess.stderr}")
        argResultImgPath.unlink(True)
        ssimValue = 101
    else:
        ssimValue = float(ssimulacraProcess.stdout.split(None, 1)[0])
    decodedPngImage.unlink(True)
    return ssimValue

def getDefaultSettings():
    jsonDict = {
        "inputFolderPath": "R:/Input",
        "resultFolderPath": "R:/",
        "resultFolderTemplate": "CropResult",
        "avifEncoderPath": "./avifenc.exe",
        "avifDecoderPath": "./avifdec.exe",
        "ssimulacraPath": "./ssimulacra2.exe",
        "jxlEncoderPath": "./cjxl.exe",
        "jxlDecoderPath": "./djxl.exe",
        "heifEncoderPath": "./heif-enc.exe",
        "heifDecoderPath": "./heif-dec.exe",
        "vipsLibPath": "./vips-dev-8.16/bin",
        "logFilePath": "./autocropLog.txt",
        "doCrop": True,
        "colorLineThreshold": 0.4,
        "colorFuzzyDistance": 16,
        "monoLineThreshold": 0.4,
        "monoFuzzyDistance": 32,
        "enableExhaustive": True,
        "exhaustiveThreshold": 5,
        "monoFuzzyDistanceEx": 32,
        "excludeVerticalCrop": True,
        "doResize": True,
        "resizeMode": 3,
        "gammaCompensation": False,
        "verticalResizeTarget": 1200,
        "horizontalResizeTarget": 1920,
        "colorThresholdPercent": 7,
        "doEncodePng": True,
        "doEncodeAvif": False,
        "doEncodeJxl": False,
        "doEncodeHeif": False,
        "pngCompressionLevel": 1,
        "avifEnableApprox": True,
        "avifForceStatistics": False,
        "avifTargetQuality": 66,
        "avifEnableBias": False,
        "avifEncodeSpeed": 6,
        "avifApproxSpeed": 6,
        "targetSsim": 80,
        "ssimTolerance": 0.5,
        "useJumpCache": True,
        "ssimToleranceInitial": 0.5,
        "jxlEncodeSpeed": 10,
        "jxlApproxSpeed": 10,
        "jxlEnableApprox": True,
        "jxlEnableBias": False,
        "jxlForceStatistics": False,
        "jxlTargetQuality": 1.5,
        "heifEnableApprox": True,
        "heifForceStatistics": False,
        "heifTargetQuality": 66,
        "heifEnableBias": False,
        "heifEncodeSpeed": "fast",
        "heifApproxSpeed": "fast" }

    return jsonDict    

def genJobList(argParamDict: dict):
    inputFolderPath = pathlib.Path(argParamDict["inputFolderPath"])

    imageFileList = [currentFile for currentFile in inputFolderPath.rglob("*") \
                        if currentFile.is_file() and currentFile.suffix in (".jpg", ".jpeg", ".png", ".jxl", ".webp")]
    
    if argParamDict["avifEnableBias"] or argParamDict["jxlEnableBias"] or argParamDict["heifEnableBias"]:
        if len(imageFileList) > multiprocessing.cpu_count():
            imageFileList.extend(imageFileList[:multiprocessing.cpu_count()])

    resultList = []
    for currentFile in imageFileList:
        currentDict = dict(argParamDict, imageFilePath=currentFile.as_posix())
        resultList.append(currentDict)
    return resultList

def loadJson(argJsonPath: pathlib.Path, argJsonType: str):
    try:
        with open(argJsonPath, "r") as jsonFile:
            jsonObject = json.load(jsonFile)
            print(f"Loaded JSON {argJsonType} file [{str(argJsonPath)}]")
            return jsonObject
    except Exception as e:
        print(f"Error reading JSON {argJsonType} file. The file may be corrupt or inaccesible.")
        print(f"Path: {argJsonPath}")
        print(f"Exact error: {str(e)}")
        exit()

def saveJson(argJsonObject, argJsonPath: pathlib.Path, argJsonType: str):
    try:
        with open(argJsonPath, "w", encoding="utf-8") as jsonFile:
            json.dump(argJsonObject, jsonFile, indent=4)
            print(f"Saved JSON {argJsonType} file [{str(argJsonPath)}].")
    except Exception as e:
        print(f"Error saving JSON {argJsonType} file.")
        print("The file may be write-protected or there is no free space. Please check and rerun the script.")
        print(f"Path: {argJsonPath}")
        print(f"Exact error: {str(e)}")

def verifyJson(argJsonObject, argCleanObject):
    errLog = []
    jsonObjectKeys = argJsonObject.keys()
    cleanObjectKeys = argCleanObject.keys()
    missingKeys = cleanObjectKeys - jsonObjectKeys
    
    if missingKeys:
        errLog.append(f"Missing keys: {missingKeys}")

    for currentKey in cleanObjectKeys:
        if type(argCleanObject[currentKey]) != type(argJsonObject[currentKey]):
            errLog.append(f"Key {currentKey} has incorrect type. Should be: {str(type(argCleanObject[currentKey]))}")

    return errLog

def saveJobJson(argJsonPath: pathlib.Path, argParams: dict):
    saveJson(genJobList(argParams), argJsonPath, "job")
    exit()

def parseArguments():
    parser = argparse.ArgumentParser(description=f"Autocrop Script 2")
    parser.add_argument("-gs", required=False, type=pathlib.Path, default=None, help="Save default settings JSON file")
    parser.add_argument("-gj", required=False, type=pathlib.Path, default=None, help="Save JSON job file")
    parser.add_argument("-ls", required=False, type=pathlib.Path, default=None, help="Load JSON settings from file")
    parser.add_argument("-lj", required=False, type=pathlib.Path, default=None, help="Load JSON job from file")
    args = parser.parse_args()
    return vars(args)

def checkDependencies(argParams: dict):
    errLog = []

    inputPath = pathlib.Path(argParams["inputFolderPath"])
    if not inputPath.exists():
        errLog.append("Error: input folder cannot be found. Check inputFolderPath in JSON.")

    resultPath = pathlib.Path(argParams["resultFolderPath"])
    if not resultPath.exists():
        errLog.append("Error: result path cannot be found.")
    
    if not vipsImportSuccess:
        vipsLibPath = pathlib.Path(argParams["vipsLibPath"])
        if not vipsLibPath.exists():
            errLog.append("Error: libVips path cannot be found and pyvips-binary can't be loaded. Check vipsLibPath in JSON.")
    
    if argParams["doEncodeAvif"]:
        avifencPath = pathlib.Path(argParams["avifEncoderPath"])
        if not avifencPath.exists():
            errLog.append("Error: Avifenc cannot be found. Check avifEncoderPath in JSON.")
        if argParams["avifEnableApprox"] or argParams["avifForceStatistics"]:
            avifdecPath = pathlib.Path(argParams["avifDecoderPath"])
            if not avifdecPath.exists():
                errLog.append("Error: Avifdec cannot be found. Check avifDecoderPath in JSON.")

    if argParams["doEncodeAvif"] or argParams["doEncodeJxl"]:
        if argParams["avifEnableApprox"] or argParams["avifForceStatistics"] or argParams["jxlEnableApprox"] or argParams["jxlForceStatistics"]:
            ssimulacraPath = pathlib.Path(argParams["ssimulacraPath"])
            if not ssimulacraPath.exists():
                errLog.append("Error: Ssimulacra2 cannot be found. Check ssimulacraPath in JSON.")

    if argParams["doEncodeJxl"]:
        cjxlPath = pathlib.Path(argParams["jxlEncoderPath"])
        if not cjxlPath.exists():
            errLog.append("Error: Cjxl cannot be found. Check jxlEncoderPath in JSON.")
        if argParams["jxlEnableApprox"] or argParams["jxlForceStatistics"]:
            djxlPath = pathlib.Path(argParams["jxlDecoderPath"])
            if not djxlPath.exists():
                errLog.append("Error: Djxl cannot be found. Check jxlDecoderPath in JSON.")

    return errLog

def main():
    startTime = time.time()

    print("Autocrop script is starting...")
    scriptDir = pathlib.Path(__file__).absolute().parent
    os.chdir(scriptDir)
    
    initSettings = parseArguments()

    if not (initSettings["gs"] or initSettings["gj"] or initSettings["ls"] or initSettings["lj"]):
        print("Non-commandline mode.")
    else:
        print("Commandline mode.")

    if initSettings["gj"] and initSettings["lj"]:
        print("Generating and loading a job JSON cannot be performed simultaneously. Exiting.")
        exit()

    if initSettings["gs"]:
        print("Saving default settings. Everything else is ignored in this mode.")
        saveJson(getDefaultSettings(), initSettings["gs"], "settings")
        exit()

    if initSettings["ls"]:
        if not initSettings["ls"].exists():
            print("Settings file specified, but not found. Exiting.")
            exit()
        jsonSettingsFile = initSettings["ls"]
    else:
        jsonSettingsFile = scriptDir.joinpath("cropDefaultSettings.json")
        if not jsonSettingsFile.exists():
            print("Default settings JSON is not found. Generating...")
            saveJson(getDefaultSettings(), jsonSettingsFile, "settings")
            exit()

    settingsDict = loadJson(jsonSettingsFile, "settings")
    errLog = verifyJson(settingsDict, getDefaultSettings())
    if len(errLog) != 0:
        print("The specified settings JSON is misconfigured. Check or regenerate it.")
        print("\n".join(errLog))
        print("Exiting.")
        exit()

    errLog = checkDependencies(settingsDict)
    if len(errLog) != 0:
        print("The specified settings JSON is misconfigured. Check or regenerate it.")
        print("\n".join(errLog))
        print("Exiting.")
        exit()

    if initSettings["gj"]:
        saveJobJson(initSettings["gj"], settingsDict)

    if not vipsImportSuccess:
        vipsLibPath = pathlib.Path(settingsDict["vipsLibPath"])
        os.environ['PATH'] = os.pathsep.join((str(vipsLibPath), os.environ['PATH']))
        global pyvips
        import pyvips

    try:
        logging.basicConfig(filename=settingsDict["logFilePath"], format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8", level=logging.DEBUG)
    except FileNotFoundError:
        print("Can't setup logging due to invalid path. Logging is disabled.")    

    logging.info("Autocrop script is starting...")
    logging.info(f"Using settings JSON ({jsonSettingsFile})")

    if not vipsImportSuccess:
        logging.info("Initial vips load failed, trying to load external library.")

    inputFolderPath = pathlib.Path(settingsDict["inputFolderPath"])
    resultFolderPath = pathlib.Path(settingsDict["resultFolderPath"])

    logging.info(f"Line threshold: color {str(settingsDict["colorLineThreshold"])}%, mono {str(settingsDict["monoLineThreshold"])}%")
    logging.info(f"Fuzzy distance: color {str(settingsDict["colorFuzzyDistance"])} colors, mono {str(settingsDict["monoFuzzyDistance"])} colors")
    logging.info(f"Input folder: {inputFolderPath}")
    logging.info(f"Output folder: {resultFolderPath}, template: {settingsDict["resultFolderTemplate"]}")

    if not settingsDict["doCrop"]:
        logging.info("Cropping is globally disabled.")
    else:
        if settingsDict["enableExhaustive"]:
            logging.info(f"Exhaustive cropping enabled, {settingsDict["exhaustiveThreshold"]}% threshold.")
        else:
            logging.info("Exhaustive cropping is globally disabled.")
    
    if not settingsDict["doResize"]:
        logging.info("Resizing is globally disabled.")

    obmDict = {}
    if settingsDict["doEncodeAvif"] or settingsDict["doEncodeJxl"] or settingsDict["doEncodeHeif"]:
        objectManager = multiprocessing.Manager()
        
    for codecPrefix in ("avif", "jxl", "heif"):
        if settingsDict["doEncode" + codecPrefix.title()]:
            if settingsDict[codecPrefix + "EnableApprox"]:
                obmDict[codecPrefix + "IterCountList"] = objectManager.list()
                obmDict[codecPrefix + "CenterPointList"] = objectManager.list()
                if settingsDict["useJumpCache"]:
                    jumpJson = scriptDir.joinpath(codecPrefix + "JumpDictCache.json")
                    if jumpJson.exists():
                        obmDict[codecPrefix + "JumpDict"] = objectManager.dict(loadJson(jumpJson, "jumpCache"))
                    else:
                        obmDict[codecPrefix + "JumpDict"] = objectManager.dict()
                if settingsDict["avifEnableBias"]:
                    obmDict[codecPrefix + "BiasList"] = objectManager.list()
            if settingsDict[codecPrefix + "EnableApprox"] or settingsDict[codecPrefix + "ForceStatistics"]:
                obmDict[codecPrefix + "SsimDiffList"] = objectManager.list()

    if initSettings["lj"]:
        if not initSettings["lj"].exists():
            print("Job file specified, but not found. Exiting.")
            exit()
        print(f"Loading job JSON ({str(initSettings["lj"])})")
        jobList = loadJson(initSettings["lj"], "job")
        for currentJob in jobList:
            currentJob.update(obmDict)
    else:
        jobList = genJobList(settingsDict | obmDict)

    maxFileCount = len(jobList)
    if maxFileCount == 0:
        print("No files to convert. Exiting.")
        logging.info("No files to convert. Exiting.")
        exit()

    logging.info(f"Converting {maxFileCount} files")
    print(f"Converting {maxFileCount} files")

    fileCounter = 0
    containsErrorsFlag = False
    with multiprocessing.Pool() as wkPool:
        wkResults = wkPool.imap_unordered(workerEntryPoint, jobList)
        for wkFile, wkDebugMessagesList, wkErrorMessagesList in wkResults:
            fileCounter += 1
            print(f"[{fileCounter}/{maxFileCount}] {wkFile}")
            for currentDebugLine in wkDebugMessagesList:   
                logging.info(currentDebugLine)
            if len(wkErrorMessagesList) > 0:
                print("Errors may have occured for this file. Check logs.")
                containsErrorsFlag = True
                for currentErrorLine in wkErrorMessagesList:
                    logging.error(currentErrorLine)
        
    logging.info("Autocrop is finished.")
    print("Autocrop is finished.")
    if containsErrorsFlag:
        print("This job may contain errors. Check logs.")

    fullRuntime = time.time() - startTime
    print(f"Total runtime: {fullRuntime}; Avg: {fullRuntime / maxFileCount}")
    logging.info(f"Total runtime: {fullRuntime}; Avg: {fullRuntime / maxFileCount}")

    for codecPrefix in ("avif", "jxl", "heif"):
        if settingsDict["doEncode" + codecPrefix.title()] and (settingsDict[codecPrefix + "EnableApprox"] or settingsDict[codecPrefix + "ForceStatistics"]):
            print("\n" + codecPrefix.title() + " statistics:")
            if settingsDict[codecPrefix + "EnableApprox"]:
                currentIterCountList = obmDict[codecPrefix + "IterCountList"].copy()
                if len(currentIterCountList) > 0:
                    print(f"Iters: Max: {max(currentIterCountList)}, Min: {min(currentIterCountList)}, Avg: {numpy.mean(currentIterCountList):.8f}")
            targetSsim = settingsDict["targetSsim"]
            currentSsimDiffList = obmDict[codecPrefix + "SsimDiffList"].copy()
            if len(currentSsimDiffList) > 0:
                print(f"Ssim: Avg: {numpy.mean(currentSsimDiffList):.8f}, Median: {numpy.median(currentSsimDiffList):.8f}, " + \
                       f"Max: {max(currentSsimDiffList)}, Min: {min(currentSsimDiffList)}, Std: {numpy.std(currentSsimDiffList, dtype=numpy.float32)}")
                ssimDiffs = [abs(x - targetSsim) for x in currentSsimDiffList]
                print(f"SSIM Diff: Max: {max(ssimDiffs):.8f}, Avg: {numpy.mean(ssimDiffs):.8f}, Median: {numpy.median(ssimDiffs):.8f}")
                print(f"SSIM Diff percentile [50, 70, 90, 99]: {numpy.percentile(ssimDiffs, [50, 70, 90, 99])}")
    
    if settingsDict["useJumpCache"]:
        for codecPrefix in ("avif", "jxl", "heif"):
            if settingsDict["doEncode" + codecPrefix.title()] and settingsDict[codecPrefix + "EnableApprox"]:
                jumpDict = obmDict[codecPrefix + "JumpDict"].copy()
                if len(jumpDict) > 0:
                    for currentKey, currentList in jumpDict.items():
                        if len(currentList) > 1:
                            if codecPrefix == "jxl":
                                jumpDict[currentKey] = [numpy.median(currentList)]
                            else:
                                jumpDict[currentKey] = [round(numpy.median(currentList))]
                    
                    saveJson(jumpDict, scriptDir.joinpath(codecPrefix + "JumpDictCache.json"), "jump cache")

    if settingsDict["doEncodeAvif"] or settingsDict["doEncodeJxl"] or settingsDict["doEncodeHeif"]:
        objectManager.shutdown()
    
if __name__ == "__main__":
    main()
