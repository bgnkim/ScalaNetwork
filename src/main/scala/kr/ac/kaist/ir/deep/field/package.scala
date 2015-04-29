package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.fn.Scalar

/**
 * Created by bydelta on 15. 4. 24.
 */
package object field {
  type Label = Int
  type VertexFeature[IN] = (Label, IN) ⇒ Scalar
  type EdgeFeature[IN] = (Label, Label, IN) ⇒ Scalar
}
