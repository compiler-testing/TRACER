module {
  func.func @main(%arg0: tensor<38x49x42x27xf32>, %arg1: tensor<38x49x1x27xi32>) -> tensor<39x49x42x27xi32> {
    %0 = "tosa.exp"(%arg0) : (tensor<38x49x42x27xf32>) -> tensor<38x49x42x27xf32>
    %1 = "tosa.clz"(%arg1) : (tensor<38x49x1x27xi32>) -> tensor<38x49x1x27xi32>
    %2 = "tosa.logical_right_shift"(%1, %1) : (tensor<38x49x1x27xi32>, tensor<38x49x1x27xi32>) -> tensor<38x49x1x27xi32>
    %3 = "tosa.argmax"(%0) {axis = 0 : i64} : (tensor<38x49x42x27xf32>) -> tensor<1x49x42x27xi32>
    %4 = "tosa.add"(%2, %3) : (tensor<38x49x1x27xi32>, tensor<1x49x42x27xi32>) -> tensor<38x49x42x27xi32>
    %5 = "tosa.div"(%1, %4) : (tensor<38x49x1x27xi32>, tensor<38x49x42x27xi32>) -> tensor<38x49x42x27xi32>
    %6 = "tosa.ceil"(%0) : (tensor<38x49x42x27xf32>) -> tensor<38x49x42x27xf32>
    %7 = "tosa.sigmoid"(%3) : (tensor<1x49x42x27xi32>) -> tensor<1x49x42x27xi32>
    %8 = "tosa.concat"(%5, %7) {axis = 0 : i64} : (tensor<38x49x42x27xi32>, tensor<1x49x42x27xi32>) -> tensor<39x49x42x27xi32>
    return %8 : tensor<39x49x42x27xi32>
  }

}