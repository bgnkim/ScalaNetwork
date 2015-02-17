organization := "kr.ac.kaist.ir"

name := "ScalaNetwork"

version := "0.1.7.0223"

scalaVersion := "2.10.4"

crossScalaVersions := Seq("2.10.4", "2.11.4")

resolvers ++= Seq("snapshots", "releases").map(Resolver.sonatypeRepo)

resolvers ++= Seq(
  "Typesafe Releases" at "http://repo.typesafe.com/typesafe/releases/"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.2.0",
  "com.typesafe.play" %% "play-json" % "2.3.4",
  "org.scalanlp" %% "breeze" % "0.10",
  "org.scalanlp" %% "breeze-natives" % "0.10",
  "org.specs2" %% "specs2-core" % "2.4.15" % "test"
)

scalacOptions in Test ++= Seq("-Yrangepos")

licenses := Seq("GNU General Public License v2" → url("http://www.gnu.org/licenses/gpl-2.0.html"))

homepage := Some(url("http://nearbydelta.github.io/ScalaNetwork"))

publishTo <<= version { v: String ⇒
  val nexus = "https://oss.sonatype.org/"
  if (v.trim.endsWith("SNAPSHOT"))
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

publishMavenStyle := true

publishArtifact in Test := false

pomIncludeRepository := { x ⇒ false}

pomExtra :=
  <scm>
    <url>git@github.com:nearbydelta/ScalaNetwork.git</url>
    <connection>scm:git:git@github.com:nearbydelta/ScalaNetwork.git</connection>
  </scm>
    <developers>
      <developer>
        <id>nearbydelta</id>
        <name>Bugeun Kim</name>
        <url>http://bydelta.kr</url>
      </developer>
    </developers>
