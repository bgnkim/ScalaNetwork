package kr.ac.kaist.ir.deep.fn

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, Serializer}

/**
 * Kryo Serializer for ScalarMatrix.
 */
object ScalarMatrixSerializer extends Serializer[ScalarMatrix] {
  /**
   * Write function
   * @param kryo Kyro object
   * @param output Output stream
   * @param t Value to be serialized
   */
  override def write(kryo: Kryo, output: Output, t: ScalarMatrix): Unit = {
    val rows = t.rows
    val cols = t.cols
    output.writeInt(rows)
    output.writeInt(cols)

    var r, c = 0
    while (r < rows) {
      while (c < cols) {
        output.writeFloat(t(r, c))
        c += 1
      }
      r += 1
    }
  }

  /**
   * Read function
   * @param kryo Kyro object
   * @param input Input stream
   * @param aClass Class of scalar matrix.
   * @return Restored Value
   */
  override def read(kryo: Kryo, input: Input, aClass: Class[ScalarMatrix]): ScalarMatrix = {
    val rows = input.readInt()
    val cols = input.readInt()

    val matrix = ScalarMatrix $0(rows, cols)
    var r, c = 0
    while (r < rows) {
      while (c < cols) {
        val f = input.readFloat()
        matrix.update(r, c, f)
        c += 1
      }
      r += 1
    }

    matrix
  }
}
