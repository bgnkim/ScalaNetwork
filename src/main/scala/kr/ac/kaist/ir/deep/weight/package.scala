package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.function._

/**
 * Package for weight
 *
 * Created by bydelta on 2014-12-30.
 */
package object weight {

  class Weight(val id: Long, private var w: Scalar = 0.0) extends Serializable {
    private var dw = 0.0

    override def toString = f"[$id] $w%.3f"

    def delta_+=(err: Scalar) = {
      dw += err
    }

    def +=(dw: Scalar) = {
      this.dw = 0.0
      w += dw
    }

    def :=(x: Scalar) = {
      dw = 0.0
      w = x
    }

    def value = w

    def delta = dw

    def +(x: Double) = w + x

    def *(x: Double) = w * x
  }

  object Weight {
    def initialize = {
      var id = 0l
      (w: () ⇒ Scalar) ⇒ {
        id += 1
        new Weight(id, w())
      }
    }
  }
}
