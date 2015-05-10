package kr.ac.kaist.ir.deep

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.util
import java.util.regex.Pattern

import kr.ac.kaist.ir.deep.fn._
import org.apache.log4j.Logger

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.io.Codec
import scala.reflect.io.{File, Path}

/**
 * Package for WordEmbedding training __(Unstable)__
 */
package object wordvec {

  /**
   * Word Filter type
   */
  trait WordFilter extends (String ⇒ String) with Serializable {
    /**
     * Tokenize given string using this filter
     * @param str String for tokenize
     * @return Array of tokens
     */
    def tokenize(str:String): mutable.WrappedArray[String]

    /**
     * Save this filter into given path
     * @param path Path to save.
     */
    def saveAs(path: Path):this.type = saveAs(File(path))

    /**
     * Save this filter into given file
     * @param file File to save.
     */
    def saveAs(file: File):this.type = {
      val oos = new ObjectOutputStream(file.outputStream())
      oos.writeObject(this)
      oos.close()
      this
    }
  }

  /** Pattern for real number **/
  final val PATTERN_REAL = Pattern.compile("^[0-9]+\\.[0-9]+$", Pattern.UNICODE_CHARACTER_CLASS)
  final val PATTERN_REAL_WITHIN = Pattern.compile("\\s+[0-9]+\\.[0-9]+\\s+", Pattern.UNICODE_CHARACTER_CLASS)
  /** Pattern for integer **/
  final val PATTERN_INTEGER = Pattern.compile("^[0-9]+$", Pattern.UNICODE_CHARACTER_CLASS)
  /** Pattern for Punctuation **/
  final val PATTERN_PUNCT = Pattern.compile("(\\p{Punct})", Pattern.UNICODE_CHARACTER_CLASS)
  /** Pattern for Special Range **/
  final val PATTERN_SPECIAL = Pattern.compile("^≪[A-Z]+≫$", Pattern.UNICODE_CHARACTER_CLASS)

  /**
   * __WordFilter__ : Filter class for take only specific language area.
   * @param langFilter Regular Expression String indicating accepted Unicode area.
   */
  case class LangFilter(langFilter: String) extends WordFilter{
    val langPattern = Pattern.compile(s"[^$langFilter\\p{Punct}]+", Pattern.UNICODE_CHARACTER_CLASS)

    /**
     * Normalize words
     * @param word Word String to be normalized
     * @return Normalized word string.
     */
    def apply(word: String) =
      if (PATTERN_SPECIAL.matcher(word).find()){
        // Remain those functional words.
        word
      } else if (PATTERN_REAL.matcher(word).find()) {
        "≪REALNUM≫"
      } else if (PATTERN_INTEGER.matcher(word).find()) {
        "≪NUMBERS≫"
      } else if (langPattern.matcher(word).find()) {
        "≪FOREIGN≫"
      } else
        word

    def tokenize(str:String): mutable.WrappedArray[String] = {
      val withReal = PATTERN_REAL_WITHIN.matcher(s" $str ")
        .replaceAll(" ≪REALNUM≫ ").trim()
      PATTERN_PUNCT.matcher(withReal).replaceAll(" $1 ").split("\\s+")
        .transform(apply)
    }
  }

  /**
   * Word2Vec model class.
   * @param map Mapping between String to Array[Coord]
   */
  class WordModel(val map: util.HashMap[String, Array[Scalar]]) extends Serializable with (String ⇒ ScalarMatrix) {
    lazy val vectorSize = map.head._2.length
    private var filter: WordFilter = LangFilter("\\u0000-\\u007f")
    private final val OTHER_VEC = map(WordModel.OTHER_UNK)

    /**
     * Set Word Filter
     * @param newFilter Filter to be set
     */
    def setFilter(newFilter: WordFilter) = {
      filter = newFilter
    }

    /**
     * Load Word Filter
     * @param path Path where Serialized Filter saved
     */
    def loadFilter(path: Path):this.type = loadFilter(File(path))

    /**
     * Load Word Filter
     * @param file File where Serialized Filter saved
     */
    def loadFilter(file: File):this.type = {
      if (file.exists && file.isFile) {
        val ois = new ObjectInputStream(file.inputStream())
        val filter = ois.readObject().asInstanceOf[WordFilter]
        ois.close()
        setFilter(filter)
      }

      this
    }

    /**
     * Get Matrix(Vector) of given word
     * @param word Word string for search
     * @return Column Vector of given word
     */
    def apply(word: String) = {
      val vec = map.getOrDefault(filter(word), OTHER_VEC)
      ScalarMatrix(vec:_*)
    }

    /**
     * Tokenize given string using word filter
     * @param str String to tokenize
     * @return Tokenized string (WrappedArray)
     */
    def tokenize(str: String) = filter.tokenize(str)

    /**
     * Tokenize given string and take average vector of them
     * @param str String to compute
     * @return Average word embedding of given string.
     */
    def tokenizeAndApply(str: String):ScalarMatrix = {
      val array = filter.tokenize(str)
      val len = array.length
      val res = ScalarMatrix $0 (vectorSize, 1)
      var i = len
      while(i > 0){
        i -= 1
        val vec = map.getOrDefault(array(i), OTHER_VEC)
        var d = vectorSize
        while(d > 0){
          d -= 1
          res(d, 0) += vec(d) / len.toFloat
        }
      }

      res
    }

    /**
     * Check existance of given word
     * @param word Word string for search
     * @return True if it is in the list
     */
    def contains(word: String) = map.containsKey(filter(word))

    /**
     * Write model into given path.
     * @param path Path where to store.
     */
    def saveAs(path: Path): Unit = saveAs(File(path))

    /**
     * Write model into given file.
     * @param file File where to store
     */
    def saveAs(file: File): Unit = {
      val bw = file.bufferedWriter(append = false, codec = Codec.UTF8)
      map.foreach {
        case (word, vec) ⇒
          bw.write(s"$word\t")
          val str = vec.map {
            v ⇒ f"$v%.8f"
          }.mkString(" ")
          bw.write(str)
      }
      bw.close()
    }
  }

  /**
   * Companion object of [[WordModel]]
   */
  object WordModel extends Serializable {
    final val OTHER_UNK = "≪UNKNOWN≫"
    val logger = Logger.getLogger(this.getClass)

    /**
     * Restore Word Model from Path.
     * @param path Path of word model file.
     * @return WordModel restored from file.
     */
    def apply(path: Path): WordModel = apply(File(path))

    /**
     * Restore WordModel from File.
     * @param file File where to read
     * @return WordModel restored from file.
     */
    def apply(file: File): WordModel =
      if (File(file.path + ".obj").exists) {
        val in = new ObjectInputStream(File(file.path + ".obj").inputStream())
        val model = in.readObject().asInstanceOf[WordModel]
        in.close()

        logger info "READ Word2Vec finished."
        model
      } else {
        val br = file.bufferedReader(Codec.UTF8)
        val firstLine = br.readLine().split("\\s+")
        val mapSize = firstLine(0).toInt
        val vectorSize = firstLine(1).toInt

        val buffer = new util.HashMap[String, Array[Scalar]]()
        var lineNo = mapSize

        while (lineNo > 0) {
          lineNo -= 1
          if (lineNo % 10000 == 0)
            logger info f"READ Word2Vec file : $lineNo%9d/$mapSize%9d"

          val line = br.readLine()
          val splits = line.split("\\s+")
          val word = splits(0)
          val vector = splits.view.slice(1, vectorSize + 1).map(_.toFloat).force
          require(vector.length == vectorSize, s"'$word' Vector is broken! Read size ${vector.length}, but expected $vectorSize")

          buffer += word → vector
        }

        br.close()

        val model = new WordModel(buffer)
        val stream = new ObjectOutputStream(File(file.path + ".obj").outputStream())
        stream.writeObject(model)
        stream.close()

        logger info "READ Word2Vec finished."
        model
      }
  }
}
