// File: app/src/main/java/com/example/facegreet/MainActivity.kt
//
// Android WiFi TCP receiver.
// Connects to the Raspberry Pi's TCP server, listens for greeting strings,
// and displays + speaks them using Text-to-Speech.
//
// Minimum SDK: 26 (Android 8.0)
// Permissions needed in AndroidManifest.xml:
//   <uses-permission android:name="android.permission.INTERNET" />

package com.example.facegreet

import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.Socket
import java.util.*

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    // ── WiFi TCP ───────────────────────────────────────────────────────────
    private val TCP_PORT = 5050
    private var tcpSocket: Socket? = null
    private var listenJob: Job? = null

    // ── TTS ────────────────────────────────────────────────────────────────
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    // ── UI ─────────────────────────────────────────────────────────────────
    private lateinit var statusText:    TextView
    private lateinit var greetingText:  TextView
    private lateinit var logView:       TextView
    private lateinit var connectButton: Button
    private lateinit var ipInput:       EditText

    private val logLines = mutableListOf<String>()

    // ──────────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText    = findViewById(R.id.statusText)
        greetingText  = findViewById(R.id.greetingText)
        logView       = findViewById(R.id.logView)
        connectButton = findViewById(R.id.connectButton)
        ipInput       = findViewById(R.id.ipInput)

        tts = TextToSpeech(this, this)

        connectButton.setOnClickListener { connectToPi() }
    }

    // ── Connect & start listening ──────────────────────────────────────────
    private fun connectToPi() {
        val host = ipInput.text.toString().trim()
        if (host.isEmpty()) {
            log("Enter the Pi's IP address.")
            return
        }

        connectButton.isEnabled = false
        statusText.text = "Connecting to $host:$TCP_PORT…"

        listenJob?.cancel()
        listenJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                tcpSocket?.close()
                tcpSocket = Socket(host, TCP_PORT)

                withContext(Dispatchers.Main) {
                    statusText.text = "Connected ✓  Listening…"
                    connectButton.isEnabled = true
                    log("Connected to $host:$TCP_PORT")
                }

                val reader = BufferedReader(InputStreamReader(tcpSocket!!.getInputStream()))
                while (isActive) {
                    val line = reader.readLine() ?: break
                    val greeting = line.trim()
                    if (greeting.isNotEmpty()) {
                        withContext(Dispatchers.Main) { onGreetingReceived(greeting) }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    log("Error: ${e.message}")
                    statusText.text = "Disconnected. Tap Connect to retry."
                    connectButton.isEnabled = true
                }
            }
        }
    }

    // ── Handle incoming greeting ───────────────────────────────────────────
    private fun onGreetingReceived(greeting: String) {
        greetingText.text = greeting
        log("Received: $greeting")
        if (ttsReady) {
            tts?.speak(greeting, TextToSpeech.QUEUE_FLUSH, null, "greeting")
        }
    }

    // ── Logging ────────────────────────────────────────────────────────────
    private fun log(msg: String) {
        val ts = java.text.SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        logLines.add("[$ts] $msg")
        if (logLines.size > 50) logLines.removeAt(0)
        logView.text = logLines.joinToString("\n")
    }

    // ── TTS init callback ──────────────────────────────────────────────────
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale.getDefault()
            ttsReady = true
        }
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────
    override fun onDestroy() {
        super.onDestroy()
        listenJob?.cancel()
        tcpSocket?.close()
        tts?.stop()
        tts?.shutdown()
    }
}
