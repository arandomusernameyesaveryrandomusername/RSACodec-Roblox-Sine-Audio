--!native
--[[
    RSC_Player — Roblox Sine Codec v6 | LocalScript  (OPTIMISED BUILD)
    =====================================================================
    Place as a LocalScript inside StarterPlayerScripts.
    Creates its own ScreenGui — no other setup needed.
    ┌── CONFIGURE THESE ──────────────────────────────┐
    │ SINE_ASSET_ID : rbxassetid of your 1-cycle      │
    │ sine wave audio asset                           │
    │ SINE_BASE_FREQ : the pitch of that asset in Hz  │
    └─────────────────────────────────────────────────┘
    RSC6 Format:
      Header : 35 bytes
      Section 1: nF * 2 * mask_sz bytes (alive + born bitmasks)
      Section 2: born_data_sz bytes (uint16 fq + uint8 amu per born partial)
      Section 3: rice_freq_sz bytes (zigzag+Rice freq deltas, MSB-first)
      Section 4: remaining bytes   (zigzag+Rice amp  deltas, MSB-first)
--]]

-- ══════════════════════════════════════════════════
-- LIBRARY LOCALS  (avoids global lookup on every call)
-- ══════════════════════════════════════════════════
local floor   = math.floor
local min     = math.min
local clamp   = math.clamp
local pow     = math.pow
local byte    = string.byte
local sub     = string.sub
local format  = string.format
local band    = bit32.band
local bor     = bit32.bor
local rshift  = bit32.rshift
local lshift  = bit32.lshift
local tcreate = table.create
local tinsert = table.insert
local tsort   = table.sort

-- ══════════════════════════════════════════════════
-- CONFIG
-- ══════════════════════════════════════════════════
local SINE_ASSET_ID  = "rbxassetid://113823829598029"
local SINE_BASE_FREQ = 1000.0

-- ══════════════════════════════════════════════════
-- SERVICES
-- ══════════════════════════════════════════════════
local Players          = game:GetService("Players")
local RunService       = game:GetService("RunService")
local SoundService     = game:GetService("SoundService")
local UserInputService = game:GetService("UserInputService")
local TweenService     = game:GetService("TweenService")
local LocalPlayer      = Players.LocalPlayer
local PlayerGui        = LocalPlayer:WaitForChild("PlayerGui")

-- ══════════════════════════════════════════════════
-- COMPILE-TIME CONSTANTS
-- ══════════════════════════════════════════════════
local INV_BASE_FREQ = 1.0 / SINE_BASE_FREQ
local FRAME_RATE    = 60

-- EQ
local EQ_COUNT = 12
local EQ_SCALE = 1200.0   -- EQ_MAX_H(100) * EQ_GAIN(3.0) — precomputed

-- ══════════════════════════════════════════════════
-- BIT-MASK LOOKUP  (avoids lshift inside tight loops)
-- BIT_MASK[bitInByte+1] where bitInByte = slot % 8
-- ══════════════════════════════════════════════════
local BIT_MASK = {1, 2, 4, 8, 16, 32, 64, 128}

-- ══════════════════════════════════════════════════
-- BINARY PRIMITIVES  (used in header parse only — one-shot)
-- ══════════════════════════════════════════════════
local function u8(s, p)
	return byte(s, p)
end
local function u16(s, p)
	local a, b = byte(s, p, p + 1)
	return a + b * 256
end
local function u32(s, p)
	local a, b, c, d = byte(s, p, p + 3)
	return a + b * 256 + c * 65536 + d * 16777216
end

-- ══════════════════════════════════════════════════
-- RSC6 HEADER PARSING
-- ══════════════════════════════════════════════════
local function parseHeader(raw)
	local magic = sub(raw, 1, 4)
	assert(magic == "RSC6", "Bad magic: " .. magic .. " (expected RSC6)")
	return {
		sampleRate  = u32(raw, 6),
		frameSize   = u32(raw, 10),
		nPartials   = u16(raw, 14),
		totalSamples= u32(raw, 16),
		totalFrames = u32(raw, 20),
		maskSz      = u16(raw, 24),
		kFreq       = u8 (raw, 26),
		kAmp        = u8 (raw, 27),
		bornDataSz  = u32(raw, 28),
		riceFreqSz  = u32(raw, 32),
	}
end

-- ══════════════════════════════════════════════════
-- RSC6 FRAME DECODER
-- Optimisations over baseline:
--   • Inlined Rice bit-reader (no per-bit function-call overhead)
--   • Sequential bitmask byte walking (no floor/rshift per slot)
--   • Inlined u16/u8 for born data
--   • MULAW_LUT replaces math.pow
-- ══════════════════════════════════════════════════
local function decodeFrames(raw, hdr)
	local nF         = hdr.totalFrames
	local nP         = hdr.nPartials
	local maskSz     = hdr.maskSz
	local kFreq      = hdr.kFreq
	local kAmp       = hdr.kAmp
	local bornDataSz = hdr.bornDataSz
	local riceFreqSz = hdr.riceFreqSz
	local fScale     = (hdr.sampleRate * 0.5) / 65535.0

	local total = nF * nP
	local freqs = tcreate(total, 0)
	local amps  = tcreate(total, 0)

	-- Section byte offsets (1-indexed Lua strings)
	local bitmaskBase  = 36
	local bornBase     = bitmaskBase + nF * 2 * maskSz
	local riceFreqBase = bornBase     + bornDataSz
	local riceAmpBase  = riceFreqBase + riceFreqSz

	-- ── Inline Rice bit-reader state ────────────────
	-- Freq reader
	local fPos      : number = riceFreqBase
	local fBuf      : number = 0
	local fBitsLeft : number = 0
	-- Amp reader
	local aPos      : number = riceAmpBase
	local aBuf      : number = 0
	local aBitsLeft : number = 0

	-- Per-slot accumulators
	local currFq  = tcreate(nP, 0)
	local currAmu = tcreate(nP, 0)

	local bornPos = bornBase  -- byte cursor into born section

	for frame = 0, nF - 1 do
		local aliveStart = bitmaskBase + frame * 2 * maskSz
		local bornStart  = aliveStart + maskSz
		local frameBase  = frame * nP

		-- Walk bitmask bytes sequentially — avoids floor/rshift per slot
		local aMaskOff = aliveStart
		local bMaskOff = bornStart
		local aByte    = 0
		local bByte    = 0
		local bitIdx   = 8  -- force reload on first slot

		for slot = 0, nP - 1 do
			-- Reload mask bytes at byte boundary
			if bitIdx == 8 then
				aByte  = byte(raw, aMaskOff)
				bByte  = byte(raw, bMaskOff)
				aMaskOff = aMaskOff + 1
				bMaskOff = bMaskOff + 1
				bitIdx = 0
			end

			local mask  = BIT_MASK[bitIdx + 1]
			local alive = band(aByte, mask) ~= 0
			bitIdx = bitIdx + 1

			local slot1 = slot + 1  -- 1-indexed into per-slot arrays

			if alive then
				local born = band(bByte, mask) ~= 0

				if born then
					-- Absolute encoding: uint16 fq + uint8 amu (inlined)
                    local ba, bb = byte(raw, bornPos, bornPos + 1)
                    local fq     = ba + bb * 256
                    local ac, ad = byte(raw, bornPos + 2, bornPos + 3)
                    local amu    = ac + ad * 256
                    bornPos = bornPos + 4
					currFq [slot1] = fq
					currAmu[slot1] = amu
				else
					-- Delta: Rice(kFreq) — fully inlined bit reader
					do
						-- Unary quotient (count leading zeros)
						local q = 0
						while true do
							if fBitsLeft == 0 then
								fBuf = byte(raw, fPos); fPos = fPos + 1; fBitsLeft = 8
							end
							fBitsLeft = fBitsLeft - 1
							if band(rshift(fBuf, fBitsLeft), 1) == 1 then break end
							q = q + 1
						end
						-- k remainder bits
						local r = 0
						for _ = 1, kFreq do
							if fBitsLeft == 0 then
								fBuf = byte(raw, fPos); fPos = fPos + 1; fBitsLeft = 8
							end
							fBitsLeft = fBitsLeft - 1
							r = lshift(r, 1) + band(rshift(fBuf, fBitsLeft), 1)
						end
						-- Zigzag decode
						local zz = lshift(q, kFreq) + r
						local delta
						if band(zz, 1) == 0 then
							delta = rshift(zz, 1)
						else
							delta = -rshift(zz + 1, 1)
						end
						currFq[slot1] = currFq[slot1] + delta
					end

					-- Delta: Rice(kAmp) — fully inlined bit reader
					do
						local q = 0
						while true do
							if aBitsLeft == 0 then
								aBuf = byte(raw, aPos); aPos = aPos + 1; aBitsLeft = 8
							end
							aBitsLeft = aBitsLeft - 1
							if band(rshift(aBuf, aBitsLeft), 1) == 1 then break end
							q = q + 1
						end
						local r = 0
						for _ = 1, kAmp do
							if aBitsLeft == 0 then
								aBuf = byte(raw, aPos); aPos = aPos + 1; aBitsLeft = 8
							end
							aBitsLeft = aBitsLeft - 1
							r = lshift(r, 1) + band(rshift(aBuf, aBitsLeft), 1)
						end
						local zz = lshift(q, kAmp) + r
						local delta
						if band(zz, 1) == 0 then
							delta = rshift(zz, 1)
						else
							delta = -rshift(zz + 1, 1)
						end
						currAmu[slot1] = clamp(currAmu[slot1] + delta, 0, 65535)
					end
				end

				freqs[frameBase + slot1] = currFq [slot1] * fScale
				amps[frameBase + slot1] = currAmu[slot1] / 65535.0
			else
				-- Dead — reset accumulators, leave freqs/amps at 0
				currFq [slot1] = 0
				currAmu[slot1] = 0
			end
		end
	end

	return freqs, amps
end

-- ══════════════════════════════════════════════════
-- SOUND POOL
-- ══════════════════════════════════════════════════
local function buildSoundPool(nPartials)
	local pool      = tcreate(nPartials)
	local container = Instance.new("Folder")
	container.Name   = "RSC_SoundPool"
	container.Parent = SoundService
	for i = 1, nPartials do
		local s = Instance.new("Sound")
		s.SoundId          = SINE_ASSET_ID
		s.Volume           = 0
		s.Looped           = true
		s.RollOffMaxDistance = 0
		s.Parent           = container
		pool[i] = s
	end
	for i = 1, nPartials do pool[i]:Play() end
	return pool
end

-- ══════════════════════════════════════════════════
-- HELPERS
-- ══════════════════════════════════════════════════
local function fmtTime(secs)
	local s = floor(secs)
	return format("%d:%02d", floor(s / 60), s % 60)
end
local function addCorner(parent, radius)
	local c = Instance.new("UICorner")
	c.CornerRadius = UDim.new(0, radius or 8)
	c.Parent = parent
	return c
end
local function addGradient(parent, c0, c1, rotation)
	local g = Instance.new("UIGradient")
	g.Color    = ColorSequence.new(c0, c1)
	g.Rotation = rotation or 0
	g.Parent   = parent
	return g
end
local function addStroke(parent, color, thickness, transparency)
	local s = Instance.new("UIStroke")
	s.Color        = color        or Color3.fromRGB(80, 200, 255)
	s.Thickness    = thickness    or 1
	s.Transparency = transparency or 0.6
	s.Parent       = parent
	return s
end
local function makeTween(obj, info, props)
	return TweenService:Create(obj, info, props)
end

-- ══════════════════════════════════════════════════
-- GUI — Premium Cyberpunk Audio Player
-- ══════════════════════════════════════════════════
local function buildGui()
	local sg = Instance.new("ScreenGui")
	sg.Name            = "RSC_PlayerGui"
	sg.ResetOnSpawn    = false
	sg.ZIndexBehavior  = Enum.ZIndexBehavior.Sibling
	sg.Parent          = PlayerGui

	local shadow = Instance.new("Frame")
	shadow.Size                  = UDim2.new(0, 520, 0, 230)
	shadow.Position              = UDim2.new(0.5, -260, 1, -248)
	shadow.BackgroundColor3      = Color3.fromRGB(0, 180, 255)
	shadow.BackgroundTransparency= 0.88
	shadow.BorderSizePixel       = 0
	shadow.ZIndex                = 1
	shadow.Parent                = sg
	addCorner(shadow, 20)

	local bg = Instance.new("Frame")
	bg.Size             = UDim2.new(0, 500, 0, 210)
	bg.Position         = UDim2.new(0.5, -250, 1, -233)
	bg.BackgroundColor3 = Color3.fromRGB(8, 10, 18)
	bg.BorderSizePixel  = 0
	bg.ZIndex           = 2
	bg.Parent           = sg
	addCorner(bg, 16)
	addGradient(bg, Color3.fromRGB(12, 14, 26), Color3.fromRGB(6, 8, 16), 120)
	addStroke(bg, Color3.fromRGB(40, 120, 220), 1.5, 0.5)

	local accentBar = Instance.new("Frame")
	accentBar.Size             = UDim2.new(1, -2, 0, 3)
	accentBar.Position         = UDim2.new(0, 1, 0, 0)
	accentBar.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
	accentBar.BorderSizePixel  = 0
	accentBar.ZIndex           = 3
	accentBar.Parent           = bg
	addCorner(accentBar, 16)
	addGradient(accentBar, Color3.fromRGB(0, 150, 255), Color3.fromRGB(80, 255, 220), 0)

	-- Left panel: EQ visualizer
	local leftPanel = Instance.new("Frame")
	leftPanel.Size             = UDim2.new(0, 130, 1, -10)
	leftPanel.Position         = UDim2.new(0, 10, 0, 5)
	leftPanel.BackgroundColor3 = Color3.fromRGB(10, 14, 24)
	leftPanel.BorderSizePixel  = 0
	leftPanel.ZIndex           = 3
	leftPanel.Parent           = bg
	addCorner(leftPanel, 12)
	addStroke(leftPanel, Color3.fromRGB(30, 80, 160), 1, 0.6)

	local EQ_WIDTH = 6
	local EQ_GAP   = 3
	local eqBars    = {}
	local eqTargets = tcreate(EQ_COUNT, 0)
	local eqTimers  = tcreate(EQ_COUNT, 0)

	local totalEqWidth = EQ_COUNT * (EQ_WIDTH + EQ_GAP) - EQ_GAP
	local eqStartX     = floor((130 - totalEqWidth) / 2)

	for i = 1, EQ_COUNT do
		local h    = math.random(10, 60)
		local xOff = eqStartX + (i - 1) * (EQ_WIDTH + EQ_GAP)

		local glow = Instance.new("Frame")
		glow.Size                  = UDim2.new(0, EQ_WIDTH + 4, 0, h + 4)
		glow.Position              = UDim2.new(0, xOff - 2, 1, -80 - h)
		glow.BackgroundColor3      = Color3.fromRGB(0, 160, 255)
		glow.BackgroundTransparency= 0.85
		glow.BorderSizePixel       = 0
		glow.ZIndex                = 4
		glow.Parent                = leftPanel
		addCorner(glow, 3)

		local bar = Instance.new("Frame")
		bar.Size             = UDim2.new(0, EQ_WIDTH, 0, h)
		bar.Position         = UDim2.new(0, xOff, 1, -78 - h)
		bar.BackgroundColor3 = Color3.fromRGB(0, 190, 255)
		bar.BorderSizePixel  = 0
		bar.ZIndex           = 5
		bar.Parent           = leftPanel
		addCorner(bar, 3)
		addGradient(bar, Color3.fromRGB(80, 230, 255), Color3.fromRGB(0, 100, 200), 90)

		eqBars[i]    = { bar = bar, glow = glow, xOff = xOff }
		eqTargets[i] = h
		eqTimers[i]  = math.random()
	end

	local codecLabel = Instance.new("TextLabel")
	codecLabel.Size               = UDim2.new(1, 0, 0, 20)
	codecLabel.Position           = UDim2.new(0, 0, 1, -28)
	codecLabel.BackgroundTransparency = 1
	codecLabel.TextColor3         = Color3.fromRGB(0, 170, 255)
	codecLabel.TextSize           = 9
	codecLabel.Font               = Enum.Font.GothamBold
	codecLabel.Text               = "RSC · SINE CODEC v6"
	codecLabel.TextXAlignment     = Enum.TextXAlignment.Center
	codecLabel.ZIndex             = 5
	codecLabel.Parent             = leftPanel

	-- Right panel
	local rightPanel = Instance.new("Frame")
	rightPanel.Size             = UDim2.new(1, -155, 1, -10)
	rightPanel.Position         = UDim2.new(0, 150, 0, 5)
	rightPanel.BackgroundTransparency = 1
	rightPanel.BorderSizePixel  = 0
	rightPanel.ZIndex           = 3
	rightPanel.Parent           = bg

	local titleRow = Instance.new("Frame")
	titleRow.Size                 = UDim2.new(1, 0, 0, 28)
	titleRow.Position             = UDim2.new(0, 0, 0, 6)
	titleRow.BackgroundTransparency = 1
	titleRow.ZIndex               = 4
	titleRow.Parent               = rightPanel

	local liveDot = Instance.new("Frame")
	liveDot.Size             = UDim2.new(0, 8, 0, 8)
	liveDot.Position         = UDim2.new(0, 0, 0.5, -4)
	liveDot.BackgroundColor3 = Color3.fromRGB(0, 255, 160)
	liveDot.BorderSizePixel  = 0
	liveDot.ZIndex           = 5
	liveDot.Parent           = titleRow
	addCorner(liveDot, 4)
	makeTween(liveDot, TweenInfo.new(0.6, Enum.EasingStyle.Sine, Enum.EasingDirection.InOut, -1, true),
		{ BackgroundTransparency = 0.85 }):Play()

	local titleLabel = Instance.new("TextLabel")
	titleLabel.Size               = UDim2.new(1, -18, 1, 0)
	titleLabel.Position           = UDim2.new(0, 14, 0, 0)
	titleLabel.BackgroundTransparency = 1
	titleLabel.TextColor3         = Color3.fromRGB(230, 240, 255)
	titleLabel.TextSize           = 14
	titleLabel.Font               = Enum.Font.GothamBold
	titleLabel.Text               = "RSC PLAYER"
	titleLabel.TextXAlignment     = Enum.TextXAlignment.Left
	titleLabel.ZIndex             = 5
	titleLabel.Parent             = titleRow

	local status = Instance.new("TextLabel")
	status.Name               = "Status"
	status.Size               = UDim2.new(1, 0, 0, 16)
	status.Position           = UDim2.new(0, 0, 0, 34)
	status.BackgroundTransparency = 1
	status.TextColor3         = Color3.fromRGB(100, 160, 220)
	status.TextSize           = 10
	status.Font               = Enum.Font.Gotham
	status.Text               = "Fetching audio.rsc …"
	status.TextXAlignment     = Enum.TextXAlignment.Left
	status.ZIndex             = 4
	status.Parent             = rightPanel

	local progressSection = Instance.new("Frame")
	progressSection.Size               = UDim2.new(1, -8, 0, 28)
	progressSection.Position           = UDim2.new(0, 0, 0, 56)
	progressSection.BackgroundTransparency = 1
	progressSection.ZIndex             = 4
	progressSection.Parent             = rightPanel

	local track = Instance.new("Frame")
	track.Size             = UDim2.new(1, 0, 0, 5)
	track.Position         = UDim2.new(0, 0, 0, 8)
	track.BackgroundColor3 = Color3.fromRGB(20, 28, 48)
	track.BorderSizePixel  = 0
	track.ZIndex           = 5
	track.Parent           = progressSection
	addCorner(track, 3)
	addStroke(track, Color3.fromRGB(30, 60, 120), 1, 0.4)

	local bar = Instance.new("Frame")
	bar.Name             = "Bar"
	bar.Size             = UDim2.new(0, 0, 1, 0)
	bar.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
	bar.BorderSizePixel  = 0
	bar.ZIndex           = 6
	bar.Parent           = track
	addCorner(bar, 3)
	addGradient(bar, Color3.fromRGB(0, 200, 255), Color3.fromRGB(60, 255, 200), 0)

	local thumb = Instance.new("Frame")
	thumb.Size             = UDim2.new(0, 12, 0, 12)
	thumb.Position         = UDim2.new(1, -6, 0.5, -6)
	thumb.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
	thumb.BorderSizePixel  = 0
	thumb.ZIndex           = 7
	thumb.Parent           = bar
	addCorner(thumb, 6)
	addGradient(thumb, Color3.fromRGB(100, 230, 255), Color3.fromRGB(0, 170, 255), 90)

	local thumbGlow = Instance.new("Frame")
	thumbGlow.Size                  = UDim2.new(0, 20, 0, 20)
	thumbGlow.Position              = UDim2.new(0, -4, 0, -4)
	thumbGlow.BackgroundColor3      = Color3.fromRGB(0, 200, 255)
	thumbGlow.BackgroundTransparency= 0.75
	thumbGlow.BorderSizePixel       = 0
	thumbGlow.ZIndex                = 6
	thumbGlow.Parent                = thumb
	addCorner(thumbGlow, 10)

	local seekBtn = Instance.new("TextButton")
	seekBtn.Name                  = "SeekBtn"
	seekBtn.Size                  = UDim2.new(1, 0, 0, 22)
	seekBtn.Position              = UDim2.new(0, 0, 0, -8)
	seekBtn.BackgroundTransparency= 1
	seekBtn.Text                  = ""
	seekBtn.ZIndex                = 8
	seekBtn.Parent                = track

	local timeRow = Instance.new("Frame")
	timeRow.Size               = UDim2.new(1, 0, 0, 16)
	timeRow.Position           = UDim2.new(0, 0, 0, 14)
	timeRow.BackgroundTransparency = 1
	timeRow.ZIndex             = 5
	timeRow.Parent             = progressSection

	local timeLabel = Instance.new("TextLabel")
	timeLabel.Name               = "TimeLabel"
	timeLabel.Size               = UDim2.new(0.5, 0, 1, 0)
	timeLabel.BackgroundTransparency = 1
	timeLabel.TextColor3         = Color3.fromRGB(120, 180, 220)
	timeLabel.TextSize           = 10
	timeLabel.Font               = Enum.Font.Code
	timeLabel.Text               = "0:00 / 0:00"
	timeLabel.TextXAlignment     = Enum.TextXAlignment.Left
	timeLabel.ZIndex             = 6
	timeLabel.Parent             = timeRow

	local partials = Instance.new("TextLabel")
	partials.Name               = "Partials"
	partials.Size               = UDim2.new(0.5, 0, 1, 0)
	partials.Position           = UDim2.new(0.5, 0, 0, 0)
	partials.BackgroundTransparency = 1
	partials.TextColor3         = Color3.fromRGB(60, 100, 160)
	partials.TextSize           = 10
	partials.Font               = Enum.Font.Code
	partials.Text               = ""
	partials.TextXAlignment     = Enum.TextXAlignment.Right
	partials.ZIndex             = 6
	partials.Parent             = timeRow

	local controlRow = Instance.new("Frame")
	controlRow.Size               = UDim2.new(1, -8, 0, 36)
	controlRow.Position           = UDim2.new(0, 0, 0, 100)
	controlRow.BackgroundTransparency = 1
	controlRow.ZIndex             = 4
	controlRow.Parent             = rightPanel

	local btn = Instance.new("TextButton")
	btn.Name             = "PlayBtn"
	btn.Size             = UDim2.new(0, 90, 1, 0)
	btn.Position         = UDim2.new(0.5, -45, 0, 0)
	btn.BackgroundColor3 = Color3.fromRGB(0, 140, 220)
	btn.TextColor3       = Color3.fromRGB(255, 255, 255)
	btn.TextSize         = 11
	btn.Font             = Enum.Font.GothamBold
	btn.Text             = "▐▐ PAUSE"
	btn.BorderSizePixel  = 0
	btn.ZIndex           = 5
	btn.Parent           = controlRow
	addCorner(btn, 8)
	addGradient(btn, Color3.fromRGB(0, 160, 255), Color3.fromRGB(0, 100, 200), 90)
	addStroke(btn, Color3.fromRGB(80, 200, 255), 1, 0.5)
	btn.MouseEnter:Connect(function()
		makeTween(btn, TweenInfo.new(0.15), { BackgroundColor3 = Color3.fromRGB(0, 180, 255) }):Play()
	end)
	btn.MouseLeave:Connect(function()
		makeTween(btn, TweenInfo.new(0.15), { BackgroundColor3 = Color3.fromRGB(0, 140, 220) }):Play()
	end)

	local volSection = Instance.new("Frame")
	volSection.Size               = UDim2.new(1, -8, 0, 28)
	volSection.Position           = UDim2.new(0, 0, 0, 145)
	volSection.BackgroundTransparency = 1
	volSection.ZIndex             = 4
	volSection.Parent             = rightPanel

	local volIcon = Instance.new("TextLabel")
	volIcon.Size               = UDim2.new(0, 22, 0, 18)
	volIcon.Position           = UDim2.new(0, 0, 0, 4)
	volIcon.BackgroundTransparency = 1
	volIcon.TextColor3         = Color3.fromRGB(60, 140, 200)
	volIcon.TextSize           = 13
	volIcon.Font               = Enum.Font.GothamBold
	volIcon.Text               = "◁)"
	volIcon.TextXAlignment     = Enum.TextXAlignment.Left
	volIcon.ZIndex             = 5
	volIcon.Parent             = volSection

	local volTrack = Instance.new("Frame")
	volTrack.Name             = "VolTrack"
	volTrack.Size             = UDim2.new(1, -80, 0, 5)
	volTrack.Position         = UDim2.new(0, 28, 0, 10)
	volTrack.BackgroundColor3 = Color3.fromRGB(20, 28, 48)
	volTrack.BorderSizePixel  = 0
	volTrack.ZIndex           = 5
	volTrack.Parent           = volSection
	addCorner(volTrack, 3)
	addStroke(volTrack, Color3.fromRGB(30, 60, 120), 1, 0.4)

	local volFill = Instance.new("Frame")
	volFill.Name             = "VolFill"
	volFill.Size             = UDim2.new(1, 0, 1, 0)
	volFill.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
	volFill.BorderSizePixel  = 0
	volFill.ZIndex           = 6
	volFill.Parent           = volTrack
	addCorner(volFill, 3)
	addGradient(volFill, Color3.fromRGB(0, 200, 255), Color3.fromRGB(60, 255, 200), 0)

	local volThumb = Instance.new("Frame")
	volThumb.Size             = UDim2.new(0, 10, 0, 10)
	volThumb.Position         = UDim2.new(1, -5, 0.5, -5)
	volThumb.BackgroundColor3 = Color3.fromRGB(200, 235, 255)
	volThumb.BorderSizePixel  = 0
	volThumb.ZIndex           = 7
	volThumb.Parent           = volFill
	addCorner(volThumb, 5)

	local volVal = Instance.new("TextLabel")
	volVal.Name               = "VolVal"
	volVal.Size               = UDim2.new(0, 40, 0, 18)
	volVal.Position           = UDim2.new(1, -40, 0, 4)
	volVal.BackgroundTransparency = 1
	volVal.TextColor3         = Color3.fromRGB(0, 190, 255)
	volVal.TextSize           = 10
	volVal.Font               = Enum.Font.GothamBold
	volVal.Text               = "100%"
	volVal.TextXAlignment     = Enum.TextXAlignment.Right
	volVal.ZIndex             = 5
	volVal.Parent             = volSection

	local volBtn = Instance.new("TextButton")
	volBtn.Name                  = "VolBtn"
	volBtn.Size                  = UDim2.new(1, 0, 0, 20)
	volBtn.Position              = UDim2.new(0, 0, 0, -7)
	volBtn.BackgroundTransparency= 1
	volBtn.Text                  = ""
	volBtn.ZIndex                = 8
	volBtn.Parent                = volTrack

	-- Slide-in animation
	bg.Position     = UDim2.new(0.5, -250, 1, 20)
	shadow.Position = UDim2.new(0.5, -260, 1, 35)
	makeTween(bg,     TweenInfo.new(0.55, Enum.EasingStyle.Back, Enum.EasingDirection.Out),
		{ Position = UDim2.new(0.5, -250, 1, -233) }):Play()
	makeTween(shadow, TweenInfo.new(0.55, Enum.EasingStyle.Back, Enum.EasingDirection.Out),
		{ Position = UDim2.new(0.5, -260, 1, -248) }):Play()

	return {
		root     = sg,
		status   = status,
		bar      = bar,
		time     = timeLabel,
		partials = partials,
		btn      = btn,
		volFill  = volFill,
		volVal   = volVal,
		volBtn   = volBtn,
		volTrack = volTrack,
		eqBars   = eqBars,
		eqTargets= eqTargets,
		eqTimers = eqTimers,
		track    = track,
		seekBtn  = seekBtn,
	}
end

-- ══════════════════════════════════════════════════
-- MAIN
-- ══════════════════════════════════════════════════
local gui = buildGui()

local function setError(msg)
	gui.status.TextColor3  = Color3.fromRGB(255, 80, 80)
	gui.status.Text        = "✕ " .. tostring(msg)
	gui.btn.Active         = false
	gui.btn.BackgroundColor3 = Color3.fromRGB(50, 20, 20)
	gui.btn.Text           = "ERROR"
	warn("[RSC_Player] " .. tostring(msg))
end

coroutine.wrap(function()
	local ok, err = pcall(function()
		gui.status.Text = "Waiting for server to fetch audio.rsc …"
		local ReplicatedStorage = game:GetService("ReplicatedStorage")
		local event = ReplicatedStorage:WaitForChild("RSC_DataEvent", 30)
		if not event then
			setError("RSC_DataEvent not found — is RSC_Fetcher.lua in ServerScriptService?")
			return
		end

		local fetchOk, raw
		event.OnClientEvent:Connect(function(ok, payload)
			fetchOk = ok
			raw     = payload
		end)

		local t = 0
		while raw == nil and t < 30 do
			task.wait(0.1); t += 0.1
		end
		if raw == nil then
			setError("Timed out waiting for RSC data from server (30s)")
			return
		end
		if not fetchOk then
			setError("Server fetch failed: " .. tostring(raw))
			return
		end

		print("[RSC DEBUG] Received bytes:", #raw)
		print("[RSC DEBUG] Magic:", sub(raw, 1, 4))
		if #raw < 35 or sub(raw, 1, 4) ~= "RSC6" then
			setError("Invalid RSC file — got " .. #raw .. " bytes, magic=["
				.. sub(raw, 1, 4) .. "] (expected RSC6)")
			return
		end

		gui.status.Text = format("Received %.1f KB — parsing header …", #raw / 1024)
		local hdrOk, hdr = pcall(parseHeader, raw)
		if not hdrOk then error("Header parse failed: " .. tostring(hdr)) end

		local nP  = hdr.nPartials
		local nF  = hdr.totalFrames
		local dur = hdr.totalSamples / hdr.sampleRate
		if nP == 0 or nF == 0 then
			error(format("Bad header: %d partials, %d frames", nP, nF))
		end

		gui.status.Text  = format("Decoding %d frames × %d partials …", nF, nP)
		gui.partials.Text= format("%d partials", nP)

		local frameFreqs, frameAmps = decodeFrames(raw, hdr)
		raw = nil  -- free memory

		gui.status.Text = format("Building %d sound objects …", nP)
		local pool = buildSoundPool(nP)

		-- ── Playback state ──────────────────────────────
		local playing       = true
		local elapsed       = 0.0
		local lastFrame     = -1
		local masterVol     = 1.0
		local prevMasterVol = 1.0   -- dirty-flag for volume cache
		local isSeeking     = false

		-- Direct audio control (no tweening)

		gui.status.Text = format("%.1fs · %d Hz · %d partials", dur, hdr.sampleRate, nP)
		gui.time.Text   = fmtTime(0) .. " / " .. fmtTime(dur)

		-- Play/Pause
		gui.btn.MouseButton1Click:Connect(function()
			playing       = not playing
			gui.btn.Text  = playing and "▐▐ PAUSE" or "▶ PLAY"
			for i = 1, nP do
				if playing then pool[i]:Resume() else pool[i]:Pause() end
			end
		end)

		-- Seek
		local function seekToPosition(x)
			local trackPos   = gui.track.AbsolutePosition.X
			local trackWidth = gui.track.AbsoluteSize.X
			elapsed = clamp((x - trackPos) / trackWidth, 0, 1) * dur
			lastFrame = -1
		end

		gui.seekBtn.MouseButton1Down:Connect(function()
			isSeeking = true
			seekToPosition(UserInputService:GetMouseLocation().X)
		end)

		-- Volume drag
		local dragging = false
		gui.volBtn.MouseButton1Down:Connect(function() dragging = true end)

		UserInputService.InputEnded:Connect(function(inp)
			if inp.UserInputType == Enum.UserInputType.MouseButton1 then
				dragging  = false
				isSeeking = false
			end
		end)

		UserInputService.InputChanged:Connect(function(inp)
			if inp.UserInputType ~= Enum.UserInputType.MouseMovement then return end
			if isSeeking then seekToPosition(inp.Position.X) end
			if dragging then
				local trackPos   = gui.volTrack.AbsolutePosition.X
				local trackWidth = gui.volTrack.AbsoluteSize.X
				local rel        = clamp((inp.Position.X - trackPos) / trackWidth, 0, 1)
				masterVol        = rel
				gui.volFill.Size = UDim2.new(rel, 0, 1, 0)
				gui.volVal.Text  = floor(rel * 100) .. "%"
			end
		end)

		-- ── EQ bucket setup ──────────────────────────────
		-- Assign each partial to one of EQ_COUNT frequency buckets
		-- based on average frequency across the song.
		local avgFreq = tcreate(nP, 0)
		local invNF   = 1.0 / nF
		for frame = 0, nF - 1 do
			local base0 = frame * nP
			for slot = 1, nP do
				avgFreq[slot] = avgFreq[slot] + frameFreqs[base0 + slot]
			end
		end
		for slot = 1, nP do avgFreq[slot] = avgFreq[slot] * invNF end

		local sortedSlots = tcreate(nP)
		for i = 1, nP do sortedSlots[i] = i end
		tsort(sortedSlots, function(a, b) return avgFreq[a] < avgFreq[b] end)

		local bucketSlots = tcreate(EQ_COUNT)
		for i = 1, EQ_COUNT do bucketSlots[i] = {} end
		for rank = 1, nP do
			local bucket = clamp(floor(rank * EQ_COUNT / nP) + 1, 1, EQ_COUNT)
			tinsert(bucketSlots[bucket], sortedSlots[rank])
		end

		-- ── Cache upvalues for HOT LOOP ──────────────────
		-- Pulling these out of the upvalue chain speeds up the
		-- Heartbeat callback significantly under --!native.
		local eqBars    = gui.eqBars
		local eqTargets = gui.eqTargets
		local guiBar    = gui.bar
		local guiTime   = gui.time
		local INV_DUR   = 1.0 / dur

		-- ── PLAYBACK LOOP ───────────────────────────────────────────────────────
		local function applyFrame(frameIdx)
			if frameIdx >= nF then frameIdx = nF - 1 end
			local base = frameIdx * nP
			local vol2x = masterVol * 2.0

			for slot = 1, nP do
				local s = pool[slot]
				local f = frameFreqs[base + slot]
				local a = frameAmps[base + slot]

				-- Directly set audio properties (no tweening)
				s.PlaybackSpeed = f * INV_BASE_FREQ
				s.Volume = a * vol2x
			end
		end

		RunService.Heartbeat:Connect(function(dt)
			-- 1. Smooth EQ bars toward their targets
			for i = 1, EQ_COUNT do
				local entry    = eqBars[i]
				local barFrame = entry.bar
				local glowFrame= entry.glow
				local xOff     = entry.xOff
				local target   = playing and eqTargets[i] or 0
				local curH     = barFrame.Size.Y.Offset
				local newH     = curH + (target - curH) * min(dt * 14, 1)
				barFrame.Size     = UDim2.new(0, 6, 0, newH)
				barFrame.Position = UDim2.new(0, xOff, 1, -78 - newH)
				glowFrame.Size    = UDim2.new(0, 10, 0, newH + 4)
				glowFrame.Position= UDim2.new(0, xOff - 2, 1, -80 - newH)
			end

			if not playing then return end

			-- 2. Advance playback
			elapsed = elapsed + dt
			if elapsed >= dur then elapsed = elapsed % dur end

			local frameIdx = floor(elapsed * FRAME_RATE)

			-- 3. Apply frame when it changes
			if frameIdx ~= lastFrame then
				applyFrame(frameIdx)
				lastFrame = frameIdx

				-- Update EQ targets from current frame
				local base = frameIdx * nP
				for i = 1, EQ_COUNT do
					local maxAmp = 0
					local slots  = bucketSlots[i]
					for s = 1, #slots do
						local a = frameAmps[base + slots[s]]
						if a > maxAmp then maxAmp = a end
					end
					eqTargets[i] = maxAmp * EQ_SCALE
				end
			end

			-- 4. Progress bar
			local progress = elapsed * INV_DUR
			guiBar.Size    = UDim2.new(progress, 0, 1, 0)
			guiTime.Text   = fmtTime(elapsed) .. " / " .. fmtTime(dur)
		end)

		Players.LocalPlayer.AncestryChanged:Connect(function()
			for i = 1, nP do
				if pool[i] and pool[i].Parent then pool[i]:Destroy() end
			end
			local container = SoundService:FindFirstChild("RSC_SoundPool")
			if container then container:Destroy() end
		end)
	end)
	if not ok then setError(err) end
end)()