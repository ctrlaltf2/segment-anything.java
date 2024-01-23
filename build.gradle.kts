plugins {
    id("java")
}

group = "dev.troyer"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    implementation("com.microsoft.onnxruntime:onnxruntime:1.16.3")
    implementation("org.nd4j:nd4j:1.0.0-M2.1")
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
}

tasks.test {
    useJUnitPlatform()
}
