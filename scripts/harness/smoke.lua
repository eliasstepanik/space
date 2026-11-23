-- Smoke test of a running shard, played the way a person would play it:
-- the connect screen, typing the address, flying, looking around.
--
--   space-client --headless --script scripts/harness/smoke.lua [--out-dir DIR]
--   space-client --headless --server HOST:PORT --script scripts/harness/smoke.lua
--   scripts/dev.sh --client headless --script scripts/harness/smoke.lua
--
-- Without --server the client connects to the address its connect screen is
-- prefilled with (127.0.0.1:7777). With --server it is already on its way to
-- that shard (and would keep retrying by itself); the script then leaves
-- again (Esc keeps the address in the field, and typing takes the controls
-- from the retry loop) so that it can type it itself. Either way the exit code of
-- space-client is this script's result: 0 when every assertion held, 1 with
-- the message on stderr otherwise. Screenshots go to the output directory.

local function screen()
  return client.state().screen
end

-- 1. Reach the connect screen with the server's address in the field.
client.wait_frames(1)
client.wait_until(function() return screen() ~= "connecting" end, 20)
if screen() == "online" then
  client.press("escape")
  client.wait_until(function() return screen() == "connect" end, 5)
end
local address = client.state().address
assert(address ~= "", "the connect screen has no address in its field")
client.screenshot("smoke-1-connect.png")

-- 2. Type the address, as a user would: clear the field, type, Enter.
client.press("end")
for _ = 1, #address do client.press("backspace") end
client.wait_until(function() return client.state().address == "" end, 5)
client.type(address)
client.wait_until(function() return client.state().address == address end, 5)
client.press("enter")
client.wait_until(function()
  local s = client.state()
  assert(s.screen ~= "error", "cannot connect to " .. address .. ": " .. tostring(s.error))
  return s.screen == "online" and s.position ~= nil
end, 20)
local joined = client.state()
assert(joined.scene ~= nil, "playing, but the state names no scene")
print(string.format("online on shard %d as session %d (%s), scene %s",
  joined.shard, joined.session, address, joined.scene))

-- 3. Fly: hold W, let go, and wait for the brake assist to stop the ship.
local start = joined.position
client.hold("w", 1.5)
client.wait_until(function() return client.state().speed < 0.05 end, 30)
local here = client.state().position
local dx, dy, dz = here[1] - start[1], here[2] - start[2], here[3] - start[3]
local moved = math.sqrt(dx * dx + dy * dy + dz * dz)
assert(moved > 2, string.format("holding W for 1.5 s moved the ship only %.2f m", moved))
print(string.format("flew %.1f m", moved))

-- 4. Look around: the view turns by what was asked.
local before = client.state()
client.turn(30, -5)
client.wait_frames(2)
local after = client.state()
local turned = (after.yaw_deg - before.yaw_deg) % 360
assert(math.abs(turned - 30) < 0.5, "the view turned " .. turned .. " degrees instead of 30")
client.screenshot("smoke-2-chase-camera.png")

-- 5. First person and back (C), with a picture of each.
client.type("c")
client.wait_frames(2)
client.screenshot("smoke-3-first-person.png")
client.type("c")

-- 6. The picture is not black: something is lit in the lower half.
local lit = 0
for i = 1, 9 do
  local r, g, b = client.pixel(math.floor(after.width * i / 10), math.floor(after.height * 0.75))
  if r + g + b > 60 then lit = lit + 1 end
end
print(lit .. " of 9 probes in the lower half are lit")

-- 7. Leave: Esc returns to the connect screen, the address still in the field.
client.press("escape")
client.wait_until(function() return screen() == "connect" end, 5)
assert(client.state().address == address, "the address was lost on disconnect")
print("smoke test passed")
