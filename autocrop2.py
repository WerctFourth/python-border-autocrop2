import numpy, numba
import os, pathlib, multiprocessing, subprocess, logging, math, uuid, json, time, argparse

#Border Autocrop 2 R2
#Requires pyVips, Numpy, Numba
#Requires external libVips library
#Can resize images using Magic Kernel 6 Sharp 7
#Run the script first time to generate JSON with settings
#resizeMode 1 - Numpy MKS67, 2 - Numpy Lanczos3, 3 - Vips Lanczos3

vipsImportSuccess = True

try:
    import pyvips
except:
    vipsImportSuccess = False
    
def getAvifCmdline(inPath: pathlib.Path, outPath: pathlib.Path, argParams: dict):
    tmpList = list()
    avifEncoderPath = pathlib.Path(argParams["avifEncoderPath"])
    tmpList.append(avifEncoderPath.as_posix())
    tmpList.extend(("-s", "3", "-q", "54", "-j", "1", "-d", "12"))
    tmpList.append(inPath.as_posix())
    tmpList.append(outPath.as_posix())
    return tmpList

def checkColor(argImg, argImgFormat: str, argParams: dict) -> bool:
    if argImgFormat == "uchar":
        currentThreshold = round(255 * argParams["colorThresholdPercent"] / 100)
        imWkArray = argImg.numpy().astype(numpy.int16)
    else:
        currentThreshold = round(65535 * argParams["colorThresholdPercent"] / 100)
        imWkArray = argImg.numpy().astype(numpy.int32)

    if numpy.max(numpy.absolute(imWkArray[:, :, 0] - imWkArray[:, :, 1])) > currentThreshold or \
            numpy.max(numpy.absolute(imWkArray[:, :, 1] - imWkArray[:, :, 2])) > currentThreshold:
        return True
    else:
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
    else:
        return numpy.dstack([resampleBand(argImgArray[:, :, i], oldSize, newSize, kernFunc, supportValue, maxColorIn, maxColorOut, linear) for i in range(bands)])

@numba.njit
def resampleBand(argImgArray, oldSize, newSize, kernFunc, supportValue, maxColorIn, maxColorOut, linear) -> numpy.ndarray:
    oldX, oldY = oldSize
    newX, newY = newSize
    argImgArrayNormal = numpy.empty_like(argImgArray, dtype=numpy.float64)
    resultArray1Stage = numpy.empty((oldY, newX), dtype=numpy.float64)
    resultArray2Stage = numpy.empty((newY, newX), dtype=numpy.uint16)

    if linear:
        for index, i in numpy.ndenumerate(argImgArray):
            argImgArrayNormal[index] = sRGBPixelValueToLinear(i, maxColorIn)
    else:
        argImgArrayNormal = argImgArray / maxColorIn
    
    weightsMap = getWeightsList(oldX, newX, supportValue, kernFunc)

    for currentYCoord in range(oldY):
        for resultPixelCoord in range(newX):
            internalSum = 0.0
            for currenSourcePixelCoord, currentWeight in weightsMap[resultPixelCoord]:
                internalSum += argImgArrayNormal[currentYCoord, currenSourcePixelCoord] * currentWeight
     
            resultArray1Stage[currentYCoord, resultPixelCoord] = internalSum

    weightsMap = getWeightsList(oldY, newY, supportValue, kernFunc)

    for currentXCoord in range(newX):
        for resultPixelCoord in range(newY):
            internalSum = 0.0
            for currenSourcePixelCoord, currentWeight in weightsMap[resultPixelCoord]:
                internalSum += resultArray1Stage[currenSourcePixelCoord, currentXCoord] * currentWeight

            if linear:
                internalSumFinal = round(linearPixelValueToSRGBNormal(internalSum) * maxColorOut)
            else:
                internalSumFinal = round(internalSum * maxColorOut)

            resultArray2Stage[resultPixelCoord, currentXCoord] = clampZero(internalSumFinal, maxColorOut) 
    
    return resultArray2Stage

@numba.njit
def getWeightsList(oldSize: int, newSize: int, supportValue: float, kernFunc) -> list[(int, float)]:
    weightsMap = list()
    scaleFactor = newSize / oldSize
    scaleFactorClamped = min(1.0, scaleFactor)
    srcWindow = supportValue / scaleFactorClamped

    for resultPixelCoord in range(newSize):
        sourceCenterPixel = (resultPixelCoord + 0.5) / scaleFactor
        firstSourcePixelCoord = math.floor(sourceCenterPixel - srcWindow)
        lastSourcePixelCoord = math.ceil(sourceCenterPixel + srcWindow)
        
        weights = [kernFunc((sourcePixelCoord + 0.5 - sourceCenterPixel) * scaleFactorClamped) for sourcePixelCoord in range(firstSourcePixelCoord, lastSourcePixelCoord + 1)]
        internalSum = sum(weights)
        weightsMap.append([(clampZero(index + firstSourcePixelCoord, oldSize - 1), cWeight / internalSum) for index, cWeight in enumerate(weights) if cWeight != 0])

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
    a = 0.055
    if s <= 0.04045:
        return s / 12.92
    return ((s+a) / (1+a)) ** 2.4

@numba.njit
def linearPixelValueToSRGB(s, maxColors):
    a = 0.055
    if s <= 0.0031308:
        return (12.92 * s) * maxColors
    return ((1+a) * s**(1/2.4) - a) * maxColors
   
@numba.njit
def linearPixelValueToSRGBNormal(s):
    a = 0.055
    if s <= 0.0031308:
        return (12.92 * s)
    return ((1+a) * s**(1/2.4) - a)

@numba.njit
def lanc3Value(x: float) -> float:
    if x > -3 and x < 3:
        return numpy.sinc(x) * numpy.sinc(x / 3)
    return 0

@numba.njit
def mkValue67(x: float) -> float:
    if x < -10 or x >= 10:
        return 0
    
    internalSum = 0.0
    for s in range(-7, 8):
        internalSum += mkSharp67Helper(s) * mk67Helper(x + s)
    return internalSum

@numba.njit
def mkSharp67Helper(offset: int) -> float:
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
def getResampleSize(imgSize: tuple[int, int], argVTarget: int, argHTarget: int) -> tuple[int, int]:
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
    if argInputColor + singleSideDistance > 255:
        leftBias = abs(argInputColor + singleSideDistance - 255)
    leftBound = max(0, argInputColor - singleSideDistance - leftBias) 
    rightBound = min(255, argInputColor + singleSideDistance + rightBias)
    return leftBound, rightBound

def cropUnif(argImageArray: numpy.ndarray, argParams: dict, argVertical: bool, argExhaustive: bool, argReverse: bool, argColor: bool):
    emptyLinesList = list()

    if argColor:
        fuzzyDistance = argParams["colorFuzzyDistance"]
        lineErrorThreshold = argParams["colorLineThreshold"]
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
        newFuzzyCount = numpy.sum(numpy.take(freq, numpy.where(numpy.logical_and(vals >= leftBound, vals <= rightBound))))
        lineFuzzyError = round((1 - newFuzzyCount / internalSize2) * 100, 4)
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
    errorMessagesList = list()
    debugMessagesList = list()

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
        if imBands == 1:
            imWkArray = numpy.copy(imArray)
        else:
            imWkArray = im.colourspace(pyvips.Interpretation.B_W).numpy()  

        topLinesList, topFuzzyError = cropUnif(imWkArray, argParams, True, False, False)
        bottomLinesList, bottomFuzzyError = cropUnif(imWkArray, argParams, True, False, True)

        if len(topLinesList) == 0:
            if topFuzzyError == -1:
                debugMessagesList.append("Fast crop (top): probably empty image")
            else:
                debugMessagesList.append(f"Fast crop (top): nothing cropped, Fe: {topFuzzyError}%")
        else:
            debugMessagesList.append(f"Fast crop (top): content line {topLinesList[-1]}, Fe: {topFuzzyError}%")
        if len(bottomLinesList) == 0:
            if bottomFuzzyError == -1:
                debugMessagesList.append("Fast crop (bottom): probably empty image")
            else:
                debugMessagesList.append(f"Fast crop (bottom): nothing cropped, Fe: {bottomFuzzyError}%")  
        else:
            debugMessagesList.append(f"Fast crop (bottom): content line {bottomLinesList[-1]}, {len(bottomLinesList)} lines, Fe: {bottomFuzzyError}%")
        imArray = numpy.delete(imArray, topLinesList + bottomLinesList, 0)

        if argParams["excludeVerticalCrop"]:
            imWkArray = numpy.delete(imWkArray, topLinesList + bottomLinesList, 0)

        leftLinesList, leftFuzzyError = cropUnif(imWkArray, argParams, False, False, False)
        rightLinesList, rightFuzzyError = cropUnif(imWkArray, argParams, False, False, True)

        if (((len(leftLinesList) + len(rightLinesList)) / imWidth) < (argParams["exhaustiveThreshold"] / 100)) and argParams["enableExhaustive"]:
            lineListH, dummyFuzzyError = cropUnif(imWkArray, argParams, False, True, False)
            debugMessagesList.append(f"Exhaustive crop (horizontal): {len(lineListH)} lines matched")
            imArray = numpy.delete(imArray, lineListH, 1)
        else:
            if len(leftLinesList) == 0:
                if leftFuzzyError == -1:
                    debugMessagesList.append("Fast crop (left): probably empty image")
                else:
                    debugMessagesList.append(f"Fast crop (left): nothing cropped, Fe: {leftFuzzyError}%")  
            else:
                debugMessagesList.append(f"Fast crop (left): content line {leftLinesList[-1]}, Fe: {leftFuzzyError}%")
            if len(rightLinesList) == 0:
                if leftFuzzyError == -1:
                    debugMessagesList.append("Fast crop (right): probably empty image")
                else:
                    debugMessagesList.append(f"Fast crop (right): nothing cropped, Fe: {rightFuzzyError}%")  
            else:
                debugMessagesList.append(f"Fast crop (right): content line {rightLinesList[-1]}, {len(rightLinesList)} lines, Fe: {rightFuzzyError}%")
            imArray = numpy.delete(imArray, leftLinesList + rightLinesList, 1)

    if argParams["doResize"]:
        oldSize = (imArray.shape[1], imArray.shape[0])
        newSize = getResampleSize(oldSize, argParams["verticalResizeTarget"], argParams["horizontalResizeTarget"])
        if oldSize == newSize:
            imResizeImg = pyvips.Image.new_from_array(imArray, interpretation="auto")
            debugMessagesList.append(f"Image is too small to resize ({oldSize[0]}x{oldSize[1]})")
        else:
            gammaCompensation = argParams["gammaCompensation"]
            match argParams["resizeMode"]:
                case 1:
                    imResizeArray = resampleImage(imArray, oldSize, newSize, imBands, imPixelSize, mkValue67, 10, gammaCompensation)
                    imResizeImg = pyvips.Image.new_from_array(imResizeArray, interpretation="auto")
                case 2:
                    imResizeArray = resampleImage(imArray, oldSize, newSize, imBands, imPixelSize, lanc3Value, 3, gammaCompensation)
                    imResizeImg = pyvips.Image.new_from_array(imResizeArray, interpretation="auto")
                case _:
                    resizeRatio = newSize[0] / oldSize[0]
                    match imColorFormat:
                        case "b-w":
                            imResizeImg = pyvips.Image.new_from_array(imArray, interpretation="auto").colourspace("gray16") \
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
        debugMessagesList.append("Encoded to PNG.")
    if argParams["doEncodeJxl"]:
        exportJxl(imResizeImg, argParams)
        debugMessagesList.append("Encoded to JXL.")
    if argParams["doEncodeAvif"]:
        exportAvif(imResizeImg, argParams, errorMessagesList, debugMessagesList)

    debugMessagesList.append(f"Local runtime: {time.time() - tmrLocal}")

    return argParams["imageFilePath"], debugMessagesList, errorMessagesList

def exportPng(argImg: numpy.ndarray, argParams: dict):
    resultFilePath = getResultFilePath("png", argParams)
    os.makedirs(resultFilePath.parent, exist_ok=True)
    argImg.write_to_file(resultFilePath, compression=argParams["pngCompressionLevel"])

def exportJxl(argImg: numpy.ndarray, argParams: dict):
    resultFilePath = getResultFilePath("jxl", argParams)
    os.makedirs(resultFilePath.parent, exist_ok=True)
    argImg.write_to_file(resultFilePath, distance=1.5, effort=8)

def exportAvif(argImg: numpy.ndarray, argParams: dict, errorList: list, debugList: list):
    resultFilePath = getResultFilePath("avif", argParams)
    os.makedirs(resultFilePath.parent, exist_ok=True)
    tmpPngFile = resultFilePath.with_name(str(uuid.uuid4()) + ".png")
    argImg.write_to_file(tmpPngFile, compression=1)
    encodeProcess = subprocess.run(getAvifCmdline(tmpPngFile, resultFilePath, argParams), \
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    if encodeProcess.returncode != 0:
        errorList.append("Possible error in AVIF encoding")
        errorList.append(f"Avifenc stderr: {encodeProcess.stderr}")
    else:
        debugList.append("Encoded to AVIF")
    tmpPngFile.unlink()

def getDefaultSettings(argKeysOnly: bool = False):
    jsonDict = {
        "inputFolderPath": "./Input",
        "resultFolderPath": "R:/",
        "resultFolderTemplate": "CropResult",
        "avifEncoderPath": "./avifenc.exe",
        "vipsLibPath": "./vips",
        "logFilePath": "./autocropLog.txt",
        "doCrop": True,
        "colorLineThreshold": 0.4,
        "colorFuzzyDistance": 16,
        "monoLineThreshold": 0.4,
        "monoFuzzyDistance": 16,
        "enableExhaustive": True,
        "exhaustiveThreshold": 5,
        "excludeVerticalCrop": True,
        "doResize": True,
        "resizeMode": 1,
        "gammaCompensation": False,
        "verticalResizeTarget": 1200,
        "horizontalResizeTarget": 1920,
        "doEncodePng": True,
        "doEncodeAvif": False,
        "doEncodeJxl": False,
        "pngCompressionLevel": 1,
        "colorThresholdPercent": 7 }
    
    if argKeysOnly:
        return jsonDict.keys()
    return jsonDict    

def genJobList(argParamDict: dict):
    inputFolderPath = pathlib.Path(argParamDict["inputFolderPath"])
    if not inputFolderPath.exists():
        print("Error! The input folder doesn't exist.")
        print(f"Path: {str(inputFolderPath)}")
        print("Exiting.")
        exit()

    imageFileList = (currentFile for currentFile in inputFolderPath.rglob("*") \
                        if currentFile.is_file() and currentFile.suffix in (".jpg", ".jpeg", ".png", ".jxl", ".webp"))

    resultList = list()
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
    print("Exiting.")
    exit()

def verifyJson(argJsonObject, argCleanObject, argJsonType: str):
    missingKeys = argCleanObject - argJsonObject.keys()
    
    if missingKeys:
        print(f"Wrong or corrupt JSON {argJsonType} file.")
        print(f"Missing keys: {missingKeys}")
        print("If it's a settings file, try to verify or regenerate it.")
        exit()

def saveDefaultSetingsJson(argJsonPath: pathlib.Path):
    print("Saving default settings. Everything else is ignored in this mode.")
    saveJson(getDefaultSettings(), argJsonPath, "settings")

def saveJobJson(argJsonPath: pathlib.Path, argJsonSettingsPath: pathlib.Path):
    if argJsonSettingsPath:
        currentSettingsPath = argJsonSettingsPath
    else:
        currentSettingsPath = pathlib.Path(__file__).absolute().parent.joinpath("cropDefaultSettings.json")
    print(f"Settings JSON file path: [{currentSettingsPath}]")
    
    if not currentSettingsPath.exists():
        print("Default or specified settings JSON is not found.")
        print("Check the specified path or regenerate the file.")
        print("Exiting.")
        exit()
    
    settingsDict = loadJson(argJsonPath, "settings")
    verifyJson(settingsDict, getDefaultSettings(True), "settings")
    jobList = genJobList(settingsDict)
    saveJson(jobList, argJsonPath, "job")

def parseArguments():
    parser = argparse.ArgumentParser(description=f"Autocrop Script 2")
    parser.add_argument("-gs", required=False, type=pathlib.Path, default=None, help="Save default settings JSON file")
    parser.add_argument("-gj", required=False, type=pathlib.Path, default=None, help="Save JSON job file")
    parser.add_argument("-ls", required=False, type=pathlib.Path, default=None, help="Load JSON settings from file")
    parser.add_argument("-lj", required=False, type=pathlib.Path, default=None, help="Load JSON job from file")
    args = parser.parse_args()

    if args.gj and args.lj:
        print("Generating and loading a job JSON cannot be performed simultaneously. Exiting.")
        exit()

    if args.gs:
        saveDefaultSetingsJson(args.gs)

    if args.ls:
        if not args.ls.exists():
            print("Settings file specified, but not found. Exiting.")
            exit()

    if args.gj:
        saveJobJson(args.gj, args.ls)

    if args.lj:
        if not args.lj.exists():
            print("Job file specified, but not found. Exiting.")
            exit()
    
    if not (args.gs or args.gj or args.ls or args.lj):
        print("Non-commandline mode.")

    return args

def main():
    startTime = time.time()

    print("Autocrop script is starting...")
    cmdArgs = parseArguments()

    if cmdArgs.ls:
        jsonSettingsFile = cmdArgs.ls
    else:
        jsonSettingsFile = pathlib.Path(__file__).absolute().parent.joinpath("cropDefaultSettings.json")
        if not jsonSettingsFile.exists():
            print("Default settings JSON is not found. Generating...")
            saveDefaultSetingsJson(jsonSettingsFile)

    settingsDict = loadJson(jsonSettingsFile, "settings")
    verifyJson(settingsDict, getDefaultSettings(True), "settings")

    if not vipsImportSuccess:
        vipsLibPath = pathlib.Path(settingsDict["vipsLibPath"])
        if not vipsLibPath.exists():
            print("Error! The libvips folder doesn't exist.")
            logging.error("The libvips folder doesn't exist. Stopping.")
            exit()  
        os.environ['PATH'] = os.pathsep.join((str(vipsLibPath), os.environ['PATH']))
        global pyvips
        import pyvips

    try:
        logging.basicConfig(filename=settingsDict["logFilePath"], format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8", level=logging.DEBUG)
    except FileNotFoundError:
        print("Can't setup logging due to invalid path. Logging is disabled.")    

    logging.info("Autocrop script is starting...")
    logging.info(f"Using settings JSON ({jsonSettingsFile})")

    inputFolderPath = pathlib.Path(settingsDict["inputFolderPath"])
    avifEncoderPath = pathlib.Path(settingsDict["avifEncoderPath"])

    if not avifEncoderPath.exists() and settingsDict["doEncodeAvif"]:
        print("Error! Avifenc is not found.")
        logging.error("AVIF encoding is enabled and Avifenc is not found. Stopping.")
        exit()     

    logging.info(f"Line threshold: color {str(settingsDict["colorLineThreshold"])}%, mono {str(settingsDict["monoLineThreshold"])}%")
    logging.info(f"Fuzzy distance: color {str(settingsDict["colorFuzzyDistance"])} colors, mono {str(settingsDict["monoFuzzyDistance"])} colors")
    logging.info(f"Input folder: {inputFolderPath}")
    logging.info(f"Output folder template: {settingsDict["resultFolderTemplate"]}")

    if not settingsDict["doCrop"]:
        logging.info("Cropping is globally disabled.")
    
    if not settingsDict["doResize"]:
        logging.info("Resizing is globally disabled.")

    if settingsDict["enableExhaustive"]:
        logging.info(f"Exhaustive cropping enabled, {settingsDict["exhaustiveThreshold"]}% threshold.")
    else:
        logging.info("Exhaustive cropping is globally disabled.")

    if cmdArgs.lj:
        print(f"Loading job JSON ({str(cmdArgs.lj)})")
        jobList = loadJson(cmdArgs.lj, "job")
    else:
        jobList = genJobList(settingsDict)

    fileCounter = 0
    maxFileCount = len(jobList)
    logging.info(f"Converting {maxFileCount} files")
    containsErrorsFlag = False

    with multiprocessing.Pool() as wkPool:
        wkResults = wkPool.imap_unordered(workerEntryPoint, jobList)
        for wkFile, wkDebugMessagesList, wkErrorMesssagesList in wkResults:
            fileCounter += 1
            print(f"[{fileCounter}/{maxFileCount}] Converting {wkFile}")
            for currentDebugLine in wkDebugMessagesList:   
                logging.info(currentDebugLine)
            if not len(wkErrorMesssagesList) == 0:
                print("Errors may have occured for this file. Check logs.")
                containsErrorsFlag = True
                for currentErrorLine in wkErrorMesssagesList:
                    logging.error(currentErrorLine)

    logging.info("Autocrop is finished.")
    print("Autocrop is finished.")
    if containsErrorsFlag:
        print("This job may contain errors. Check logs.")

    fullRuntime = time.time() - startTime
    print(f"Total runtime: {fullRuntime}; Avg: {fullRuntime / maxFileCount}")
    logging.info(f"Total runtime: {fullRuntime}; Avg: {fullRuntime / maxFileCount}")
    
if __name__ == "__main__":
    main()
