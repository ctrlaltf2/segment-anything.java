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
    implementation("com.microsoft.onnxruntime:onnxruntime:1.14.0")
}

tasks.test {
    useJUnitPlatform()
}
