--[[
    RSC_Decoder.lua  —  LocalScript (place in StarterGui or StarterPlayer.StarterPlayerScripts)
    
    Decodes RSC2 audio in real-time using a pooled sine wave synthesizer.
    Features:
    - Efficient binary parsing (bit32 for u16/f32)
    - Sound pool (reuses 192 looped sine sounds)
    - Real-time frame-by-frame playback
    - Simple GUI with progress/controls
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local UserInputService = game:GetService("UserInputService")
local RunService = game:GetService("RunService")
local SoundService = game:GetService("SoundService")
local tcreate = table.create

-- ════════════════════════════════════════════════════════════════════════════
-- CONSTANTS & CONFIG
-- ════════════════════════════════════════════════════════════════════════════

local RSC_MAGIC = "RSC2"
local HEADER_FMT = "<BBIHHHIIf"
local FRAME_HDR_SIZE = 6  -- f32(4) + u16(2)
local PARTIAL_SIZE = 6    -- u16(2) + u16(2) + u16(2)

local MAX_PARTIALS = 192
local SINE_POOL_SIZE = MAX_PARTIALS
local SINE_ASSET_ID = "rbxassetid://113823829598029"

-- ════════════════════════════════════════════════════════════════════════════
-- BINARY READING HELPERS
-- ════════════════════════════════════════════════════════════════════════════

local function readU16BE(data, off)
	local b1 = string.byte(data, off)
	local b2 = string.byte(data, off + 1)
	return bit32.bor(bit32.lshift(b1, 8), b2)
end

local function readU16LE(data, off)
	local b1 = string.byte(data, off)
	local b2 = string.byte(data, off + 1)
	return bit32.bor(b1, bit32.lshift(b2, 8))
end

local function readI16LE(data, off)
	local u16 = readU16LE(data, off)
	if u16 >= 32768 then
		return u16 - 65536
	end
	return u16
end

local function readF32LE(data, off)
	local b1 = string.byte(data, off)
	local b2 = string.byte(data, off + 1)
	local b3 = string.byte(data, off + 2)
	local b4 = string.byte(data, off + 3)

	local bits = bit32.bor(
		b1,
		bit32.lshift(b2, 8),
		bit32.lshift(b3, 16),
		bit32.lshift(b4, 24)
	)

	-- IEEE 754 single precision decode
	local sign = bit32.rshift(bits, 31)
	local exp = bit32.band(bit32.rshift(bits, 23), 0xFF)
	local mant = bit32.band(bits, 0x7FFFFF)

	if exp == 0 then
		return 0
	elseif exp == 255 then
		return math.huge
	end

	local value = 1 + mant / 0x800000
	value = value * math.pow(2, exp - 127)

	if sign == 1 then
		value = -value
	end

	return value
end

-- ════════════════════════════════════════════════════════════════════════════
-- SINE SOUND POOL
-- ════════════════════════════════════════════════════════════════════════════

local function buildSoundPool(nSlots)
	local pool      = tcreate(nSlots)
	local container = Instance.new("Folder")
	container.Name   = "RSC2_SoundPool"
	container.Parent = SoundService

	for i = 1, nSlots do
		local s = Instance.new("Sound")
		s.SoundId            = SINE_ASSET_ID
		s.Volume             = 0
		s.Looped             = true
		s.RollOffMaxDistance = 0
		s.Parent             = container
		pool[i] = s
	end

	-- Wait for asset to load on the first sound instance
	if not pool[1].IsLoaded then
		local loaded = false
		local connection
		connection = pool[1].Loaded:Connect(function() 
			loaded = true 
			connection:Disconnect()
		end)

		local t = 0
		while not loaded and t < 10 do
			task.wait(0.1)
			t += 0.1
		end
	end

	-- Start all sounds in the pool (muted)
	for i = 1, nSlots do 
		pool[i]:Play() 
	end

	return pool, container
end

local SinePool = {}
SinePool.__index = SinePool

function SinePool.new(parent, size, sample_rate)
	local self = setmetatable({}, SinePool)
	self.sounds, self.container = buildSoundPool(size)
	self.freqs = {}
	self.amps = {}
	self.phases = {}
	self.active = {}
	self.bin_to_slot = {}
	self.slot_to_bin = {}
	self.sample_rate = sample_rate
	self.size = size

	for i = 1, size do
		self.freqs[i] = 1000
		self.amps[i] = 0
		self.phases[i] = 0
		self.active[i] = false
		self.slot_to_bin[i] = nil
	end
	return self
end

function SinePool:_setSound(idx, freq, amp, phase)
	self.freqs[idx] = freq
	self.amps[idx] = amp
	self.phases[idx] = phase

	local sound = self.sounds[idx]
	local speed = math.max(freq / 1000, 0.01)
	sound.PlaybackSpeed = speed
	sound.Volume = amp

	local phase_norm = (phase / (2 * math.pi)) % 1
	sound.TimePosition = phase_norm / 1000
end

function SinePool:assign(bin, freq, amp, phase)
	local idx = self.bin_to_slot[bin]

	if idx and self.active[idx] then
		-- Update existing partial
		self:_setSound(idx, freq, amp, phase)
		return idx
	end

	-- Allocate a free slot
	for i = 1, self.size do
		if not self.active[i] then
			idx = i
			break
		end
	end

	if not idx then
		idx = 1
		local old_bin = self.slot_to_bin[idx]
		if old_bin then
			self.bin_to_slot[old_bin] = nil
		end
	end

	self.active[idx] = true
	self.bin_to_slot[bin] = idx
	self.slot_to_bin[idx] = bin

	self:_setSound(idx, freq, amp, phase)
	return idx
end

function SinePool:release(idx)
	local old_bin = self.slot_to_bin[idx]
	if old_bin then
		self.bin_to_slot[old_bin] = nil
		self.slot_to_bin[idx] = nil
	end

	self.active[idx] = false
	self.amps[idx] = 0
	self.sounds[idx].Volume = 0
end

function SinePool:releaseUnused(active_bins)
	for i = 1, self.size do
		if self.active[i] then
			local bin = self.slot_to_bin[i]
			if not active_bins[bin] then
				self:release(i)
			end
		end
	end
end

function SinePool:update(delta_time)
	for i = 1, self.size do
		if self.active[i] then
			self.sounds[i].Volume = self.amps[i]
		else
			self.sounds[i].Volume = 0
		end
	end
end

-- ════════════════════════════════════════════════════════════════════════════
-- RSC2 DECODER
-- ════════════════════════════════════════════════════════════════════════════

local RSC2 = {}
RSC2.__index = RSC2

function RSC2.new(rsc_data, parent)
	local self = setmetatable({}, RSC2)

	-- Parse header
	if string.sub(rsc_data, 1, 4) ~= RSC_MAGIC then
		error("Invalid RSC2 magic")
	end

	self.data = rsc_data
	self.offset = 5

	-- Read header (little-endian)
	self.version = string.byte(rsc_data, 5)
	self.channels = string.byte(rsc_data, 6)
	self.sample_rate = readU16LE(rsc_data, 7) + (bit32.lshift(readU16LE(rsc_data, 9), 16))
	self.fft_size = readU16LE(rsc_data, 11)
	self.hop_size = readU16LE(rsc_data, 13)
	self.max_partials = readU16LE(rsc_data, 15)
	self.n_frames = readU16LE(rsc_data, 17) + (bit32.lshift(readU16LE(rsc_data, 19), 16))
	self.n_samples = readU16LE(rsc_data, 21) + (bit32.lshift(readU16LE(rsc_data, 23), 16))
	self.window_sum = readF32LE(rsc_data, 25)

	self.offset = 29

	-- Create sine pool
	self.pool = SinePool.new(parent, SINE_POOL_SIZE, self.sample_rate)

	-- Playback state
	self.current_frame = 0
	self.is_playing = false
	self.playback_start_time = 0

	print(string.format("[RSC2] v%d | %d Hz | %d ch | %d frames | FFT=%d hop=%d",
		self.version, self.sample_rate, self.channels, self.n_frames, self.fft_size, self.hop_size))

	return self
end

function RSC2:readFrame(frame_idx, ch)
	-- Seek to frame data
	-- File layout: [header(29)] [ch0_frame0] [ch0_frame1] ... [ch1_frame0] ...
	local frame_off = 29 + (ch * self.n_frames + frame_idx) * (FRAME_HDR_SIZE + self.max_partials * PARTIAL_SIZE)

	-- Read frame header
	local peak = readF32LE(self.data, frame_off)
	local n_p = readU16LE(self.data, frame_off + 4)

	local partials = {}
	local off = frame_off + FRAME_HDR_SIZE

	for i = 1, n_p do
		local bin = readU16LE(self.data, off)
		local amp_u16 = readU16LE(self.data, off + 2)
		local phase_u16 = readU16LE(self.data, off + 4)

		local amp = (amp_u16 / 65535.0) * peak / self.window_sum
		local phase = (phase_u16 / 65535.0) * 2 * math.pi - math.pi

		local freq = (bin * self.sample_rate) / self.fft_size

		table.insert(partials, {
			bin = bin,
			freq = freq,
			amp = amp,
			phase = phase
		})

		off = off + PARTIAL_SIZE
	end

	return partials
end

function RSC2:playFrame(frame_idx)
	if frame_idx >= self.n_frames then
		self.is_playing = false
		return
	end

	-- Read first channel for now (mono playback)
	local partials = self:readFrame(frame_idx, 0)
	local active_bins = {}

	for _, partial in ipairs(partials) do
		active_bins[partial.bin] = true
		self.pool:assign(partial.bin, partial.freq, partial.amp, partial.phase)
	end

	self.pool:releaseUnused(active_bins)
	self.current_frame = frame_idx
end

function RSC2:play()
	self.is_playing = true
	self.playback_start_time = tick()
	self:playFrame(0)
end

function RSC2:stop()
	self.is_playing = false
	for i = 1, self.pool.size do
		self.pool:release(i)
	end
end

function RSC2:update()
	if not self.is_playing then
		return
	end

	-- Calculate which frame we should be on
	local elapsed = tick() - self.playback_start_time
	local frame_idx = math.floor((elapsed * self.sample_rate) / self.hop_size)

	if frame_idx >= self.n_frames then
		self:stop()
		return
	end

	if frame_idx > self.current_frame then
		self:playFrame(frame_idx)
	end

	-- Update sine pool
	self.pool:update(0.016)  -- Assume 60 FPS
end

-- ════════════════════════════════════════════════════════════════════════════
-- GUI
-- ════════════════════════════════════════════════════════════════════════════

local function createGUI(decoder)
	local screenGui = Instance.new("ScreenGui")
	screenGui.Name = "RSC_DecoderGUI"
	screenGui.ResetOnSpawn = false
	screenGui.Parent = game.Players.LocalPlayer:WaitForChild("PlayerGui")

	-- Background
	local bg = Instance.new("Frame")
	bg.Name = "Background"
	bg.Size = UDim2.new(0, 300, 0, 150)
	bg.Position = UDim2.new(0.5, -150, 0.5, -75)
	bg.BackgroundColor3 = Color3.fromRGB(30, 30, 30)
	bg.BorderSizePixel = 0
	bg.Parent = screenGui

	-- Title
	local title = Instance.new("TextLabel")
	title.Name = "Title"
	title.Size = UDim2.new(1, 0, 0, 30)
	title.BackgroundColor3 = Color3.fromRGB(20, 20, 20)
	title.TextColor3 = Color3.fromRGB(100, 200, 255)
	title.TextSize = 14
	title.Font = Enum.Font.GothamBold
	title.Text = "RSC2 Decoder"
	title.BorderSizePixel = 0
	title.Parent = bg

	-- Status
	local status = Instance.new("TextLabel")
	status.Name = "Status"
	status.Size = UDim2.new(1, -10, 0, 25)
	status.Position = UDim2.new(0, 5, 0, 35)
	status.BackgroundTransparency = 1
	status.TextColor3 = Color3.fromRGB(200, 200, 200)
	status.TextSize = 12
	status.Font = Enum.Font.Gotham
	status.TextXAlignment = Enum.TextXAlignment.Left
	status.Text = "Stopped"
	status.Parent = bg

	-- Progress
	local progress = Instance.new("TextLabel")
	progress.Name = "Progress"
	progress.Size = UDim2.new(1, -10, 0, 20)
	progress.Position = UDim2.new(0, 5, 0, 65)
	progress.BackgroundTransparency = 1
	progress.TextColor3 = Color3.fromRGB(150, 150, 150)
	progress.TextSize = 10
	progress.Font = Enum.Font.Gotham
	progress.Text = "Frame: 0 / 0"
	progress.Parent = bg

	-- Play button
	local playBtn = Instance.new("TextButton")
	playBtn.Name = "PlayButton"
	playBtn.Size = UDim2.new(0, 80, 0, 30)
	playBtn.Position = UDim2.new(0, 10, 0, 95)
	playBtn.BackgroundColor3 = Color3.fromRGB(50, 150, 50)
	playBtn.TextColor3 = Color3.fromRGB(255, 255, 255)
	playBtn.TextSize = 12
	playBtn.Font = Enum.Font.GothamBold
	playBtn.Text = "▶ Play"
	playBtn.BorderSizePixel = 0
	playBtn.Parent = bg

	playBtn.MouseButton1Click:Connect(function()
		decoder:play()
	end)

	-- Stop button
	local stopBtn = Instance.new("TextButton")
	stopBtn.Name = "StopButton"
	stopBtn.Size = UDim2.new(0, 80, 0, 30)
	stopBtn.Position = UDim2.new(0, 100, 0, 95)
	stopBtn.BackgroundColor3 = Color3.fromRGB(150, 50, 50)
	stopBtn.TextColor3 = Color3.fromRGB(255, 255, 255)
	stopBtn.TextSize = 12
	stopBtn.Font = Enum.Font.GothamBold
	stopBtn.Text = "⏹ Stop"
	stopBtn.BorderSizePixel = 0
	stopBtn.Parent = bg

	stopBtn.MouseButton1Click:Connect(function()
		decoder:stop()
	end)

	-- Close button
	local closeBtn = Instance.new("TextButton")
	closeBtn.Name = "CloseButton"
	closeBtn.Size = UDim2.new(0, 80, 0, 30)
	closeBtn.Position = UDim2.new(0, 190, 0, 95)
	closeBtn.BackgroundColor3 = Color3.fromRGB(80, 80, 80)
	closeBtn.TextColor3 = Color3.fromRGB(255, 255, 255)
	closeBtn.TextSize = 12
	closeBtn.Font = Enum.Font.GothamBold
	closeBtn.Text = "✕ Close"
	closeBtn.BorderSizePixel = 0
	closeBtn.Parent = bg

	closeBtn.MouseButton1Click:Connect(function()
		screenGui:Destroy()
	end)

	-- Update loop
	RunService.RenderStepped:Connect(function()
		decoder:update()

		if decoder.is_playing then
			status.Text = "▶ Playing..."
		else
			status.Text = "Stopped"
		end

		progress.Text = string.format("Frame: %d / %d", decoder.current_frame, decoder.n_frames)
	end)

	return screenGui
end

-- ════════════════════════════════════════════════════════════════════════════
-- MAIN
-- ════════════════════════════════════════════════════════════════════════════

local function main()
	print("[RSC_Decoder] Waiting for RSC data from server...")

	local event = ReplicatedStorage:WaitForChild("RSC_DataEvent")
	local success, rsc_data = event.OnClientEvent:Wait()

	if not success then
		warn("[RSC_Decoder] Server error:", rsc_data)
		return
	end

	print("[RSC_Decoder] Received " .. #rsc_data .. " bytes")

	-- Create decoder
	local decoder = RSC2.new(rsc_data, workspace)

	-- Create GUI
	createGUI(decoder)

	print("[RSC_Decoder] Ready! Click Play to start.")
end

main()