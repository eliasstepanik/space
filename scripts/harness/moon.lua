-- Fast travel from the demo area to the Moon and a landing on it, played
-- the way a person would: switch fast travel on (F), aim at the Moon, hold
-- W; switch it off near the Moon, come down braking, stand on the ground.
--
--   space-client --headless --server HOST:PORT --script scripts/harness/moon.lua \
--       --script-timeout 300 --out-dir DIR
--
-- The exit code of space-client is this script's result. Screenshots go to
-- the output directory: moon-1-spawn, moon-2-approach, moon-3-landed.

local function sky(name)
  for _, b in ipairs(client.state().sky) do
    if b.name == name then return b end
  end
  error("no " .. name .. " in the sky")
end

local function sub(a, b) return { a[1] - b[1], a[2] - b[2], a[3] - b[3] } end
local function len(v) return math.sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3]) end

-- Turn the view towards scene point `p` (yaw left+, pitch up+, degrees).
local function aim(p)
  local s = client.state()
  local d = sub(p, s.scene_position)
  local yaw = math.deg(math.atan(-d[1], -d[3]))
  local pitch = math.deg(math.atan(d[2], math.sqrt(d[1] * d[1] + d[3] * d[3])))
  local dyaw = (yaw - s.yaw_deg + 540) % 360 - 180
  client.turn(dyaw, pitch - s.pitch_deg)
end

-- 1. On the shard, in the demo area, Earth the nearest body.
client.wait_frames(1)
if client.state().screen ~= "online" then
  client.connect(client.state().address, 20)
end
client.wait_until(function() return client.state().scene_position ~= nil end, 20)
local s = client.state()
assert(s.scene == "demo-area", "expected the demo area, got " .. tostring(s.scene))
assert(s.nearest == "Earth", "nearest at the spawn: " .. tostring(s.nearest))
assert(s.gravity < 0.01, "the spawn point is weightless")
client.screenshot("moon-1-spawn.jpg")

-- 2. Fast travel on; steer at the Moon until it is a few kilometres away.
client.type("f")
client.wait_until(function() return client.state().fast end, 5)
local started = client.state().frame
local top = 0
client.key_down("w")
client.wait_until(function()
  aim(sky("Moon").position)
  local st = client.state()
  if st.speed > top then top = st.speed end
  return sky("Moon").surface_m < 3000
end, 150, 0.2)
client.key_up("w")
client.log(string.format("at the Moon: %.0f m over it, top speed %.0f km/s",
  sky("Moon").surface_m, top / 1000))
assert(top > 1.0e6, "fast travel was not fast: " .. top)
client.screenshot("moon-2-approach.jpg")

-- 3. Fast travel off (the ship drops to its normal top speed), brake, then
--    down: push towards the ground while slower than a twentieth of the
--    height per second, let the brake assist work otherwise.
client.type("f")
client.wait_until(function() return not client.state().fast end, 5)
client.wait_until(function() return client.state().speed < 1 end, 60)
client.wait_until(function()
  local st = client.state()
  local moon = sky("Moon")
  local h = moon.surface_m
  if h < 1.5 or st.landed then
    client.key_up("shift")
    return st.landed
  end
  -- "down" is towards the Moon's centre: look at it, then push forward.
  aim(moon.position)
  if st.speed < math.max(1, h / 20) then
    client.key_down("w")
  else
    client.key_up("w")
  end
  return false
end, 200, 0.1)
client.key_up("w")
client.wait(2)
s = client.state()
assert(s.nearest == "Moon", "landed on " .. tostring(s.nearest))
assert(s.landed, "not standing on the Moon")
assert(s.gravity > 1.0, "the Moon's pull at the ship: " .. s.gravity)
assert(s.nearest_surface_m < 3, "over the ground: " .. s.nearest_surface_m)
client.log(string.format("landed: gravity %.2f m/s2, %.1f m over the ground, speed %.2f",
  s.gravity, s.nearest_surface_m, s.speed))
-- Look along the ground, at Earth.
client.turn(0, -30)
client.wait(0.5)
client.screenshot("moon-3-landed.jpg")
client.log("frames " .. (client.state().frame - started))
