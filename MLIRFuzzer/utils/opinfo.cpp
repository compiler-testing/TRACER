//
// Created by Administrator on 2023/3/4.
//
#include "TosaGen/opinfo.h"
#include "TosaGen/utils.h"
#include "TosaGen/transfer.h"
#include "TosaGen/create.h"

extern Utils genUtils;
extern InfoGen infogen;
extern opInfo info;
extern Transfer transfer;
extern Create create;
SmallVector<int, 8> perms_const_int;
string table_input2type;
int pad_input1_rank;

opInfo initStruct(opInfo info){
  info.opName = "";
  info.inputNum = 0;
  info.attsNum = 0;
  info.needBroadcast  = 0;

  info.inputType= {};
  info.attrs = {};
  info.resultType = nullptr;
  info.inputs = {};
  info.Constraint = {};
  return info;
}

/*
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
using namespace mlir;
using namespace mlir::tosa;
#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaInterfaces.cpp.inc"
*/

void InfoGen::initInfo(string opname){
  /*根据opname初始化opinfo*/
//  cout<<"initializing"<<endl;
  auto opConstraint = genUtils.getConstraint(opname);
  info = initStruct(info);
  info.opName = opConstraint["OP_NAME"].asString();
  info.Constraint = opConstraint;
  SmallVector<StringRef , 8> opConNames ={"concat","add","bitwise_and","bitwise_or","bitwise_xor","div","equal","greater","greater_equal","logical_and","logical_left_shift","logical_or","logical_right_shift","logical_xor","maximum","minimum","pow","sub","arithmetic_right_shift","mul","select"}; //,"cond_if"
  auto isConOp = std::find(opConNames.begin(), opConNames.end(), info.opName);
  if(isConOp != opConNames.end())
    info.needBroadcast=1;
  else
    info.needBroadcast=0;

  info.attsNum = opConstraint["ATTR_NUM"].asInt();
  info.inputNum = opConstraint["INPUT_NUM"].asInt();

//  if(info.opName=="select")
//   info.inputNum = 2;
}

// std::vector<int> randperm(int Num)
// {
// 	std::vector<int> temp;
// 	for (int i = 0; i < Num; ++i)
// 	{
// 		temp.push_back(i);
// 	}
// 	random_shuffle(temp.begin(), temp.end());
//   return temp;
// }

SmallVector<SmallVector<int64_t,8>,8> genInputShape() {
//  Json::Value inputs = info.Constraint["INPUT"];
  int inputNum = info.inputNum;

  int Dim;
  llvm::SmallVector<int64_t, 8> shape1;
  llvm::SmallVector<int64_t, 8> shape2;
  SmallVector<SmallVector<int64_t, 8>, 8> inputShapes;

  SmallVector<SmallVector<int64_t,8>,8> curOpDim = genUtils.getDimFromopConstraint();

  SmallVector<int64_t,8> cShape; //ConstraintShape
  for (int i = 0; i < curOpDim.size();
       i++) { // 循环遍历“INPUT”中的多个对象，获取每个对象中的维度大小
    cShape = curOpDim[i];
    if (cShape.size() > 1) {
      int rDim = genUtils.genRandomN(0, cShape.size() - 1);
      Dim = cShape[rDim];
    } else if (cShape.size() == 1) {
      if (cShape[0] == 100 && info.opName != "select"){
          Dim = genUtils.genRandomN(0, 6);
          if(info.opName == "pad"){
              pad_input1_rank = Dim;
              //cout<<"pad_input1_rank    "<<pad_input1_rank<<endl;
          }
      }

      else if(cShape[0] == 100 && info.opName == "select"){
          Dim = genUtils.genRandomN(1, 6);
      }
      else
        Dim = cShape[0];
    }
    if (i == 1 && info.needBroadcast == 1) {
      shape2 = genUtils.genConstrainedShape(shape1);
      inputShapes.push_back(shape2);
    } else {
      if(i==0 && !info.inputType.empty()) {
        shape1 = genUtils.getTensorShape(info.inputType[0]);
      }else{
        shape1 = genUtils.genShape(i,Dim,inputShapes); // 生成形状

      }
      inputShapes.push_back(shape1);
    }

  }
  // if(info.needBroadcast == 1){
  //   int v = genUtils.genRandomN(0,inputShapes[0].size()-1);
  //   std::vector<int> vec = randperm(inputShapes[0].size());

  //   for(int i=0;i<inputShapes.size();i++){
  //     for(int j =0;j<v;j++){
  //       inputShapes[i][vec[j]]=mlir::ShapedType::kDynamic;
  //     }
  //   }
  // }
  return inputShapes;
}

SmallVector<string,8> genInputType(){
  Json::Value inputs = info.Constraint["INPUT"];
  int inputNum = info.inputNum;
  string typeStr;
  string elementType;
  SmallVector<string, 8> ElementTypes;
  for (int i = 0;i < inputNum; i++) {
    if(i==0 && !info.inputType.empty()){
      elementType = genUtils.getTensorType(info.inputType[0]);
      ElementTypes.push_back(elementType);
    }else{
      typeStr = inputs[i]["TYPE"].asString();
      if (info.opName == "conv2d" || info.opName == "conv3d" ||
          info.opName == "transpose_conv2d" ||
          info.opName == "depthwise_conv2d") {
        typeStr = "conv";
      }
      if(info.opName == "select" && i == 0){
          typeStr = "i1";
      }
      if (info.opName == "avg_pool2d" || info.opName == "max_pool2d") {
        typeStr = "anyNoI1";
      }
      // 操作数元素相同的情况
      string firstType;
      if (typeStr == "any" || typeStr == "float" || typeStr == "int" ||
          typeStr == "floatOruint" || typeStr == "anyNoI1" || typeStr == "conv") {
        if (i == 0 || (info.opName == "select" && i == 1))
          elementType = genUtils.getTypestr(typeStr);
        ElementTypes.push_back(elementType);
      } else {
        elementType = genUtils.getTypestr(typeStr);
        if(info.opName == "table" && i ==1){
            if(elementType == "i8"){
                table_input2type = "i8";
            }else if(elementType == "i16"){
                table_input2type = "i16";
            }
        }

        ElementTypes.push_back(elementType);

      }
    }
  }

  return ElementTypes;
}

void InfoGen::addInputType(ImplicitLocOpBuilder b){
//  cout<<"addInputType     "<<info.inputNum<<endl;
  SmallVector<string,8> elementTypes = genInputType();
  SmallVector<SmallVector<int64_t,8>,8> inputShapes = genInputShape();
  Type tensorType;
  for(int i = 0;i<inputShapes.size();i++) {
    tensorType = genUtils.genTensorType(b, inputShapes[i], elementTypes[i]);
    info.inputType.push_back(tensorType);
  }
//  genUtils.printTypes(info.inputType);
}

void InfoGen::addInputType(ImplicitLocOpBuilder b,Value preOp){
//  cout << "addInputType" << endl;
  info.inputType.push_back(preOp.getType());
  if(info.inputNum>1){
    Json::Value inputs = info.Constraint["INPUT"];
    SmallVector<SmallVector<int64_t, 8>, 8> inputShapes = genInputShape();
    SmallVector<string,8> elementTypes = genInputType();

    Type tensorType;
    for(int i = 1;i<inputShapes.size();i++) {
      tensorType = genUtils.genTensorType(b, inputShapes[i], elementTypes[i]);
      info.inputType.push_back(tensorType);
    }
  }
//  genUtils.printTypes(info.inputType);
}/*
void buildConvOpWithQuantInfo(OpBuilder &builder, ImplicitLocOpBuilder b,
                              Type outputType, Value input, Value weight,
                              Value bias) {
    auto opConstraint = info.Constraint;
    Json::Value attrs = opConstraint["ATTR"];
    SmallVector<NamedAttribute> namedAttrs;
    auto quantAttr = buildConvOpQuantizationAttr(builder, input, weight);
    if (attrs[0]["NAME"].asString() == "quantization_info") {
//        result.addAttribute("quantization_info", quantAttr);
//        result.addTypes(
//                buildConvOpResultTypeInfo(builder, outputType, input, weight));
//    } else {
//        result.addTypes(outputType);
//    }
        namedAttrs.push_back(b.getNamedAttr("quantization_info", quantAttr));
}*/

extern int concat_axis;
void InfoGen::addAttrs(ImplicitLocOpBuilder b, Location loc){
//  cout<<"addAttrs"<<endl;
  int attsNum = info.attsNum;
  auto opConstraint = info.Constraint;
  Json::Value attrs = opConstraint["ATTR"];
  SmallVector<NamedAttribute> namedAttrs;

  if(attsNum==1) {
    if (attrs[0]["NAME"].asString() == "axis") {
      int Dim = info.inputType[0].cast<ShapedType>().getRank();
      int axis = genUtils.genRandomN(0, Dim - 1);

      if (info.opName == "concat"){
//        int axis=0;
        SmallVector<int> index;
        if(info.inputType.size()==2){
          auto i1= info.inputType[0].cast<ShapedType>().getShape();
          auto i2= info.inputType[1].cast<ShapedType>().getShape();
          if(i1.size()==i2.size()){
            for(int i = 0;i<i2.size();i++){
              if(i1[i]!=i2[i]){
                index.push_back(i);
              }
            }
          }
        }
        if(!index.empty()){
        if(index.size()==1)
          axis=index[0];
        else
          axis = index[genUtils.genRandomN(0,index.size()-1)];
        }
      }
//      cout<<"axis  : "<< axis<<endl;
      namedAttrs.push_back(b.getNamedAttr("axis", b.getI64IntegerAttr(axis)));
    } else if (attrs[0]["NAME"].asString() == "new_shape") {
      SmallVector<int64_t, 8> newShape;
      int en = genUtils.getElementNum(
          info.inputType[0].cast<ShapedType>().getShape());
      // 判断是不是素数
      int prime = 0;
      for (int i = 2; i < en; i++) {
        if (en % i == 0)
          prime++;
      }
      // 如果元素个数是素数，包含1
      if (prime == 0) {
        newShape.push_back(en);
        int size = genUtils.genRandomN(1, 3);
        while (size > 0) {
          newShape.push_back(1);
          size--;
        }
      } else {
        int size = genUtils.genRandomN(1, 4);

        SmallVector<int64_t, 8> factors = genUtils.getPrimeFactors(en);
        random_shuffle(factors.begin(), factors.end());
        if (size == 1) {
          newShape.push_back(en);
        } else {
          int temp_size = 1;
          int product = 1;

          while (temp_size <= size) {
            if (factors.size() == 0) {
              newShape.push_back(1);
            } else if (temp_size == size) {
              product = 1;
              while (factors.size() > 0) {
                product = product * factors.back();
                factors.pop_back();
              }
              newShape.push_back(product);
            } else {
              int slice = genUtils.genRandomN(0, factors.size() - 1);
              SmallVector<int64_t, 8> slice_vec;
              product = 1;
              for (; slice > 0; slice--) {
                product = product * factors.back();
                factors.pop_back();
              }
              newShape.push_back(product);
            }
            temp_size++;
          }
        }
      }
      random_shuffle(newShape.begin(), newShape.end());
      namedAttrs.push_back(b.getNamedAttr("new_shape",b.getDenseI64ArrayAttr(newShape)));
//      namedAttrs.push_back(
//          b.getNamedAttr("new_shape", b.getI64ArrayAttr(newShape)));
    }else if (attrs[0]["NAME"].asString() == "round"){
        srand((time(0)));
        auto round = rand()%2;
        namedAttrs.push_back(b.getNamedAttr("round", b.getBoolAttr(round)));
    }else if(attrs[0]["NAME"].asString() == "shift"){
        srand((time(0)));
        auto shift = rand()%256 - 128;
        namedAttrs.push_back(b.getNamedAttr("shift", b.getI32IntegerAttr(shift)));
    }/*else if(attrs[0]["NAME"].asString() == ""){

    }*/
    else if(attrs[0]["NAME"].asString() == "multiples"){
        SmallVector<int64_t, 8> multiple;
        info.inputType[0].dump();
        for(int num = 0;num < info.inputType[0].cast<ShapedType>().getRank();num++){
//            auto rank = info.inputType[0].cast<ShapedType>().getRank();
//            cout << rank<<endl;
//            auto shape = info.inputType[0].cast<ShapedType>().getShape();
//            cout << shape[0]<<endl;
            int size = genUtils.genRandomN(1, 3);
            multiple.push_back(size) ;
        }
        namedAttrs.push_back(b.getNamedAttr("multiples", b.getDenseI64ArrayAttr(multiple)));
        for (auto i :multiple)
            cout<<i<<endl;

    }

  }else{
    int attrArrayCount;
    string attrName;
    //std::uniform_int_distribution<unsigned> randomBig(0, 126549185);
    //  std::uniform_int_distribution<unsigned> randomSmall(1, 2);
    if (info.opName=="conv2d" || info.opName=="conv3d" || info.opName=="depthwise_conv2d"
        || info.opName=="max_pool2d" || info.opName=="avg_pool2d" || info.opName=="transpose_conv2d" || info.opName=="pad"){
      for (int i = 0; i < attsNum; i++) {
        SmallVector<int64_t , 8> newShape ;
        attrName = attrs[i]["NAME"].asString();
        attrArrayCount = attrs[i]["ArrayCount"][0].asInt();
        if (attrName == "quantization_info"){  //对于属性 "quantization_info"，先暂时不处理
          break;
        }
        if (attrName == "out_shape"){
          attrArrayCount = 4;
        }
        for (int j = 0; j < attrArrayCount; j++) {
          unsigned int r =  genUtils.genRandomN(1,2);
          newShape.push_back(r);

        }
        namedAttrs.push_back(b.getNamedAttr(attrName,b.getDenseI64ArrayAttr(newShape)));
      }
    }

    else if (info.opName=="slice"){
        attrArrayCount = info.inputType[0].cast<ShapedType>().getRank();
        for(int i=0;i<attsNum;i++){
            SmallVector<int64_t , 8> newShape;
            attrName = attrs[i]["NAME"].asString();
            if (attrName == "start"){
                for (int j = 0; j < attrArrayCount; j++) {
                        auto shape = info.inputType[0].cast<ShapedType>().getShape();
                        unsigned int k =  genUtils.genRandomN(1,shape[i]);
                                            newShape.push_back(k);
                }
            }
            else if(attrName == "size"){
                for (int j = 0; j < attrArrayCount; j++) {
                    unsigned int k =  genUtils.genRandomN(1,12);
                    newShape.push_back(k);
                }
            }
            namedAttrs.push_back(b.getNamedAttr(attrName,b.getDenseI64ArrayAttr(newShape)));
        }
    }
    else if (info.opName=="clamp"){
        for(int i=0;i<attsNum;i++){
            attrName = attrs[i]["NAME"].asString();
            srand((time(0)));
            long long min_int_Shape = rand()%128-64;
            float min_fp_Shape = rand()%128-64;
            if (attrName == "min_int"){
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getI64IntegerAttr(min_int_Shape)));
            }else if(attrName == "max_int"){
                long long max_int_Shape = rand()%128+min_int_Shape;
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getI64IntegerAttr(max_int_Shape)));
            }else if(attrName == "min_fp"){
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getF32FloatAttr(min_fp_Shape)));
            }else if(attrName == "max_fp"){
                float max_fp_Shape = rand()%128+min_int_Shape;
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getF32FloatAttr(max_fp_Shape)));
            }
        }

    }

    else if (info.opName == "resize"){
        for(int i=0;i<attsNum;i++){
            SmallVector<int64_t , 8> newShape;
            if (attrName == "scale"){
                for(int i=0;i<4;i++){
                    newShape.push_back(genUtils.genRandomN(1,100));
                }
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getDenseI64ArrayAttr(newShape)));
            }else if (attrName == "offset"){
                for(int i=0;i<2;i++){
                    newShape.push_back(genUtils.genRandomN(0,0));
                }
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getDenseI64ArrayAttr(newShape)));
            }else if (attrName == "border"){
                for(int i=0;i<2;i++){
                    newShape.push_back(genUtils.genRandomN(0,0));
                }
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getDenseI64ArrayAttr(newShape)));
            }else if(attrName == "mode"){
                string newShape1 =  "BILINEAR";
               namedAttrs.push_back(b.getNamedAttr(attrName, b.getStringAttr(newShape1)));
            }

        }
    }
    else if (info.opName=="rescale"){
        auto rank=genUtils.genRandomN(1,4);
        for(int i=0;i<attsNum;i++){
            attrName = attrs[i]["NAME"].asString();
            if(attrName == "input_zp"){
                auto input_zp=genUtils.genRandomN(1,256);
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getI32IntegerAttr(input_zp)));
            }else if(attrName == "output_zp"){
                auto output_zp=genUtils.genRandomN(1,256);
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getI32IntegerAttr(output_zp)));
            }else if(attrName == "multiplier"){
                SmallVector<int32_t , 8> newShape;
                for (int j = 0; j < rank; j++) {
                    unsigned int k =  genUtils.genRandomN(1,64);
                    newShape.push_back(k);
                }
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getDenseI32ArrayAttr(newShape)));
            }else if(attrName == "shift"){
                SmallVector<int32_t , 8> newShape;
                for (int j = 0; j < rank; j++) {
                    unsigned int k =  genUtils.genRandomN(1,64);
                    newShape.push_back(k);
                }
                namedAttrs.push_back(b.getNamedAttr(attrName,b.getDenseI32ArrayAttr(newShape)));
            }else if(attrName == "scale32"){
                srand((time(0)));
                auto scale32 = rand()%2;
                namedAttrs.push_back(b.getNamedAttr(attrName, b.getBoolAttr(scale32)));
            }else if(attrName == "double_round"){
                auto double_round = rand()%2;
                namedAttrs.push_back(b.getNamedAttr(attrName, b.getBoolAttr(double_round)));
            }else if(attrName == "per_channel"){
                auto per_channel = rand()%2;
                namedAttrs.push_back(b.getNamedAttr(attrName, b.getBoolAttr(per_channel)));
            }
        }
    }
  }
  info.attrs = namedAttrs;
}



Value genDependentInput(ImplicitLocOpBuilder b, Location loc, Value first_input){
    auto input1_size = first_input.getType().cast<ShapedType>().getRank();
    llvm::SmallVector<int64_t , 8> perms_shape = {input1_size};

    //随机选择perms值的类型,生成DenseIntOrFPElementsAttr
    string typeStr = genUtils.getTypestr("Int32Or64");
    DenseIntOrFPElementsAttr inputAttr;
    if (typeStr=="i32"){
        llvm::SmallVector<int32_t , 8> perms_const;
        //生成perms参数的值  i32
        // 创建从0开始递增的数组
        for (int i = 0; i < input1_size; i++) {
            perms_const.push_back(i);
        }
        // 打乱顺序
        std::random_shuffle(perms_const.begin(), perms_const.end());
        //保存perms_const用于后续result推断
        for(auto v : perms_const){
            perms_const_int.push_back(v);
        }
        //生成DenseIntElementsAttr  i32
        inputAttr = DenseIntElementsAttr::get(RankedTensorType::get(perms_shape, b.getI32Type()), perms_const);
    }else if (typeStr=="i64") {
        llvm::SmallVector<int64_t , 8> perms_const;
        //生成perms参数的值  i64
        // 创建从0开始递增的数组
        for (int i = 0; i < input1_size; i++) {
            perms_const.push_back(i);
        }
        // 打乱顺序
        std::random_shuffle(perms_const.begin(), perms_const.end());
        //保存perms_const用于后续result推断
        for(auto v : perms_const){
            perms_const_int.push_back(v);
        }
        //生成DenseIntElementsAttr  i64
        inputAttr = DenseIntElementsAttr::get(RankedTensorType::get(perms_shape, b.getI64Type()), perms_const);
    }
    //创建常量
    Value constOp = b.create<tosa::ConstOp>(loc, inputAttr.getType(), inputAttr);
    return constOp;
}

void InfoGen::addInputs(ImplicitLocOpBuilder b, Location loc,func::FuncOp funcOp) {
  cout<<"addInputs"<<endl;
  auto opConstraint = info.Constraint;
  SmallVector<Value, 8> argsValue; //获取Value， 保存至info。inputs 作为op的输入
  int startindex =  funcOp.getNumArguments()-info.inputNum;
  for (int j = startindex; j < funcOp.getNumArguments(); j++) {
    argsValue.push_back(funcOp.getArgument(j));
  }
  info.inputs =argsValue;


  //transpose 根据第一个输入生成第二个输入
  if(info.opName=="transpose"){
      Value perms = genDependentInput(b,loc,info.inputs[0]);
      argsValue.push_back(perms);
      info.inputType.push_back(perms.getType());
      info.inputs.push_back(perms);
  }
  genUtils.printValues(info.inputs);
}

void InfoGen::addInputs(ImplicitLocOpBuilder b, Location loc,func::FuncOp funcOp,Value preOp) {
//  cout<<"addInputs"<<endl;
  auto opConstraint = info.Constraint;
  SmallVector<Value, 8> argsValue;
  SmallVector<Type, 8> argTypes;
  info.inputs.push_back(preOp);
  if(info.inputNum>1){//创建输入
    int startIndex = funcOp.getNumArguments();
    argTypes = info.inputType;
    argTypes.erase(argTypes.begin());
    create.insertFuncArg(b,funcOp,argTypes); //插入index

    for (int j = startIndex; j < funcOp.getNumArguments(); j++) {
      info.inputs.push_back(funcOp.getArgument(j));
    }
  }

    //transpose 根据第一个输入生成第二个输入
    if(info.opName=="transpose"){
        Value perms = genDependentInput(b,loc,info.inputs[0]);
        argsValue.push_back(perms);
        info.inputType.push_back(perms.getType());
        info.inputs.push_back(perms);
    }

    genUtils.printValues(info.inputs);
}


void InfoGen::addResult(mlir::ImplicitLocOpBuilder b) {
  cout<<"addResult:   ";
  pair<string, SmallVector<int64_t, 8>> result =  transfer.transferResult(b);
  Type resultType = genUtils.genTensorType(b, result.second, result.first);
  info.resultType = resultType;
  info.resultType.dump();
}


pair<string, SmallVector<int64_t, 8>> InfoGen::getResultInfo(mlir::Value op) {

  auto type = op.getType().cast<ShapedType>().getElementType();
  auto shape = op.getType().cast<ShapedType>().getShape();
  pair<string, SmallVector<int64_t, 8>> resultInfo = {genUtils.type2str(type), genUtils.getShapeVector(shape)};
  return  resultInfo;
}