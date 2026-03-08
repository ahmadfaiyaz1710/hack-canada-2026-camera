// File: app/src/main/java/com/example/facegreet/MainActivity.kt
//
// Android Bluetooth RFCOMM receiver.
// Pairs with the Raspberry Pi, listens for greeting strings,
// and displays + speaks them using Text-to-Speech.
//
// Minimum SDK: 26 (Android 8.0)
// Permissions needed in AndroidManifest.xml:
//   <uses-permission android:name="android.permission.BLUETOOTH" />
//   <uses-permission android:name="android.permission.BLUETOOTH_ADMIN" />
//   <uses-permission android:name="android.permission.BLUETOOTH_CONNECT"
//                    android:maxSdkVersion="30" />   <!-- API 31+ -->
//   <uses-permission android:name="android.permission.BLUETOOTH_SCAN"
//                    android:maxSdkVersion="30" />   <!-- API 31+ -->

package com.example.facegreet

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.*
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.*

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    // ── Bluetooth ─────────────────────────────────────────────────────────────
    // Standard SPP (Serial Port Profile) UUID — must match the Pi's RFCOMM server
    private val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")

    private var btAdapter: BluetoothAdapter? = null
    private var btSocket: BluetoothSocket? = null
    private var listenJob: Job? = null

    // ── TTS ───────────────────────────────────────────────────────────────────
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    // ── UI ────────────────────────────────────────────────────────────────────
    private lateinit var statusText:    TextView
    private lateinit var greetingText:  TextView
    private lateinit var logView:       TextView
    private lateinit var connectButton: Button
    private lateinit var deviceSpinner: Spinner

    private val logLines = mutableListOf<String>()

    // ─────────────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText    = findViewById(R.id.statusText)
        greetingText  = findViewById(R.id.greetingText)
        logView       = findViewById(R.id.logView)
        connectButton = findViewById(R.id.connectButton)
        deviceSpinner = findViewById(R.id.deviceSpinner)

        tts = TextToSpeech(this, this)
        btAdapter = BluetoothAdapter.getDefaultAdapter()

        if (btAdapter == null) {
            statusText.text = "Bluetooth not available on this device."
            return
        }

        requestBtPermissions()
        populatePairedDevices()

        connectButton.setOnClickListener { connectToSelectedDevice() }
    }

    // ── Permissions ───────────────────────────────────────────────────────────
    private fun requestBtPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.BLUETOOTH_CONNECT,
                    Manifest.permission.BLUETOOTH_SCAN
                ),
                1
            )
        }
    }

    // ── Populate spinner with already-paired devices ──────────────────────────
    private fun populatePairedDevices() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
            != PackageManager.PERMISSION_GRANTED &&
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.S
        ) return

        val paired: Set<BluetoothDevice> = btAdapter?.bondedDevices ?: emptySet()
        val names = paired.map { it.name ?: it.address }.toMutableList()
        if (names.isEmpty()) names.add("No paired devices found")

        deviceSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, names)
            .also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
    }

    // ── Connect & start listening ─────────────────────────────────────────────
    private fun connectToSelectedDevice() {
        val selectedName = deviceSpinner.selectedItem?.toString() ?: return
        val device = btAdapter?.bondedDevices?.firstOrNull {
            (it.name ?: it.address) == selectedName
        } ?: run {
            log("Device not found.")
            return
        }

        connectButton.isEnabled = false
        statusText.text = "Connecting to $selectedName…"

        listenJob?.cancel()
        listenJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                btSocket?.close()
                btSocket = device.createRfcommSocketToServiceRecord(SPP_UUID)
                btSocket!!.connect()

                withContext(Dispatchers.Main) {
                    statusText.text = "Connected ✓  Listening…"
                    connectButton.isEnabled = true
                    log("Connected to $selectedName")
                }

                // Read lines from the socket indefinitely
                val reader = BufferedReader(InputStreamReader(btSocket!!.inputStream))
                while (isActive) {
                    val line = reader.readLine() ?: break   // null = socket closed
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

    // ── Handle incoming greeting ──────────────────────────────────────────────
    private fun onGreetingReceived(greeting: String) {
        greetingText.text = greeting
        log("Received: $greeting")
        if (ttsReady) {
            tts?.speak(greeting, TextToSpeech.QUEUE_FLUSH, null, "greeting")
        }
    }

    // ── Logging ───────────────────────────────────────────────────────────────
    private fun log(msg: String) {
        val ts = java.text.SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        logLines.add("[$ts] $msg")
        if (logLines.size > 50) logLines.removeAt(0)
        logView.text = logLines.joinToString("\n")
    }

    // ── TTS init callback ─────────────────────────────────────────────────────
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale.getDefault()
            ttsReady = true
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    override fun onDestroy() {
        super.onDestroy()
        listenJob?.cancel()
        btSocket?.close()
        tts?.stop()
        tts?.shutdown()
    }
}
