package kr.ac.kaist.ir.deep.wordvec

import java.util

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.apache.log4j._

import scala.collection.JavaConversions._

/**
 * Train Word2Vec and save the model.
 */
object PrepareCorpus extends Logging {
  {
    // Initialize Network Logging
    val PATTERN = "%d{yy/MM/dd HH:mm:ss} %p %C{2}: %m%n"
    val orgFile = new RollingFileAppender(new PatternLayout(PATTERN), "spark.log")
    orgFile.setMaxFileSize("1MB")
    orgFile.setMaxBackupIndex(5)
    val root = Logger.getRootLogger
    root.addAppender(orgFile)
    root.setLevel(Level.WARN)
    root.setAdditivity(false)
    val krFile = new RollingFileAppender(new PatternLayout(PATTERN), "trainer.log")
    krFile.setMaxFileSize("1MB")
    krFile.setMaxBackupIndex(10)
    val kr = Logger.getLogger("kr.ac")
    kr.addAppender(krFile)
    kr.setLevel(Level.INFO)
  }

  /**
   * Main thread.
   * @param args CLI arguments
   */
  def main(args: Array[String]) =
    if (args.length == 0 || args.contains("--help") || args.contains("-h")) {
      println(
        """Tokenize sentences, and Collect several types of unknown words.
          |
          |== Arguments without default ==
          | -i	Path of input corpora file.
          | -o	Path of tokenized output text file.
          |
          |== Arguments with default ==
          | --srlz	Local Path of Serialized Language Filter file. (Default: filter.dat)
          | --thre	Minimum include count. (Default: 3)
          | --part	Number of partitios. (Default: organized by Spark)
          | --lang	Accepted Language Area of Unicode. (Default: \\\\u0000-\\\\u007f)
          |       	For Korean: 가-힣|\\\\u0000-\\\\u007f
          |
          |== Additional Arguments ==
          | --help	Display this help message.
          | """.stripMargin)
    } else {
      // Set spark context
      val conf = new SparkConf()
        .setAppName("Normalize Infrequent words")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.scheduler.mode", "FAIR")
        .set("spark.shuffle.memoryFraction", "0.05")
        .set("spark.storage.unrollFraction", "0.05")
        .set("spark.storage.memoryFraction", "0.9")
        .set("spark.broadcast.blockSize", "40960")
        .set("spark.akka.frameSize", "50")
        .set("spark.locality.wait", "10000")
      val sc = new SparkContext(conf)
      sc.setLocalProperty("spark.scheduler.pool", "production")

      val langArea = getArgument(args, "--lang", "\\u0000-\\u007f")
      val langFilter = LangFilter(langArea)
      val bcFilter = sc.broadcast(langFilter)
      langFilter.saveAs(getArgument(args, "--srlz", "filter.dat"))
      logInfo(s"Language filter created : $langArea")

      // read file
      val in = getArgument(args, "-i", "article.txt")
      val parts = getArgument(args, "--part", "1").toInt
      val lines = sc.textFile(in, parts).filter(_.trim.nonEmpty)
      val tokens = tokenize(lines, bcFilter)

      val threshold = getArgument(args, "--thre", "3").toInt
      val infreqWords = infrequentWords(tokens.flatMap(x ⇒ x), threshold)
      val infreqSet = sc.broadcast(infreqWords)

      val out = getArgument(args, "-o", "article-preproc.txt")
      normalizedTokens(tokens, infreqSet).saveAsTextFile(out)

      // Stop the context
      sc.stop()
    }

  /**
   * Read argument
   * @param args Argument Array
   * @param key Argument Key
   * @param default Default value of this argument
   * @return Value of this key.
   */
  def getArgument(args: Array[String], key: String, default: String) = {
    val idx = args.indexOf(key)
    if (idx < 0 || idx > args.length - 1) default
    else args(idx + 1)
  }

  /**
   * Collect frequent words with count >= Threshold
   * @param words Word seq.
   * @return HashSet of frequent words.
   */
  def infrequentWords(words: RDD[String], threshold: Int) = {
    val counts = words.countByValue()
    val above = counts.count(_._2 >= threshold)
    val set = counts.filter(_._2 < threshold).keySet
    val value = new util.HashSet[String]()
    value ++= set

    val all = above + set.size
    val ratio = Math.round(set.size.toFloat / all * 100)
    logInfo(s"Total $all distinct words, ${set.size} words($ratio%) will be discarded.")

    value
  }

  /**
   * Convert input into tokenized string, using Stanford NLP toolkit.
   * @param lines Input lines
   * @return tokenized & normalized lines.
   */
  def tokenize(lines: RDD[String], bcFilter: Broadcast[_ <: WordFilter]) =
    lines.map(bcFilter.value.tokenize).persist(StorageLevel.DISK_ONLY_2)

  /**
   * Convert tokenized string into a sentence, with appropriate conversion of (Threshold - 1) count word.
   * @param input Tokenized input sentence
   * @param infreqSet Less Frequent words
   * @return Tokenized converted sentence
   */
  def normalizedTokens(input: RDD[_ <: Seq[String]], infreqSet: Broadcast[util.HashSet[String]]) =
    input.mapPartitions {
      lazy val set = infreqSet.value

      _.map {
        seq ⇒
          val it = seq.iterator
          val buf = StringBuilder.newBuilder

          while(it.hasNext){
            val word = it.next()
            if (set contains word){
              buf.append(WordModel.OTHER_UNK)
            }else{
              buf.append(word)
            }
            buf.append(' ')
          }

          buf.result()
      }
    }
}
