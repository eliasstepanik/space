-- A tour of the demo area's physics showcase, played the way a person would
-- play it, that asserts the physics happened:
--
--   * in the gravity room the ship falls onto the floor and stands there
--     (`landed`, gravity 9.81 m/s2), and its lift engines (Space) lift it
--     off again;
--   * flying into a spring pendulum sets it swinging;
--   * in the zero-g corner a floating crate the ship flies into drifts off,
--     and the door swings open when the ship pushes it;
--   * on the planetoid (large-body gravity) the ship lands on top, feels its
--     pull, and lifts off again; its moons keep circling it.
--
--   space-client --headless --server HOST:PORT --script scripts/harness/physics.lua \
--       --script-timeout 300 [--out-dir DIR]
--   scripts/dev.sh --client headless --script scripts/harness/physics.lua
--
-- It needs a shard showing the demo-area scene with its physics (phase 2 of
-- the physics work). Its exit code is the result; the screenshots go to the
-- output directory. The tour moves things for good (a pushed crate drifts
-- off in weightless space), as any player would.

local function state()
  return client.state()
end

local function at()
  return client.state().scene_position
end

local function distance(a, b)
  local x, y, z = a[1] - b[1], a[2] - b[2], a[3] - b[3]
  return math.sqrt(x * x + y * y + z * z)
end

local function fmt(p)
  return string.format("(%.2f %.2f %.2f)", p[1], p[2], p[3])
end

-- The body `guid`, as the client draws it now (nil when it is not drawn).
local function body(guid)
  for _, b in ipairs(state().bodies) do
    if b.guid == guid then
      return b
    end
  end
  return nil
end

-- The nearest body of `visual` for which `keep(b)` holds.
local function find(visual, keep)
  for _, b in ipairs(state().bodies) do
    if b.visual == visual and (keep == nil or keep(b)) then
      return b
    end
  end
  return nil
end

-- Turn the view towards the horizontal direction (dx, dz), level.
local function face(dx, dz)
  local s = state()
  local want = math.deg(math.atan(-dx, -dz))
  local turn = (want - s.yaw_deg + 540) % 360 - 180
  if math.abs(turn) > 0.5 or math.abs(s.pitch_deg) > 0.5 then
    client.turn(turn, -s.pitch_deg)
  end
end

local held = {}
local function hold(key, down)
  if down and not held[key] then
    client.key_down(key)
    held[key] = true
  elseif not down and held[key] then
    client.key_up(key)
    held[key] = false
  end
end

local function let_go()
  for _, key in ipairs({ "w", "s", "a", "d", "space", "shift" }) do
    hold(key, false)
  end
end

-- Hold `more` while `speed` is well below `want`, `less` while it is well
-- above, neither in between.
local function steer(speed, want, more, less, slack)
  hold(more, speed < want - slack)
  hold(less, speed > want + slack)
end

-- Fly to `target` (metres from the spawn point, the frame of
-- `scene_position`), as a pilot would with the keys: face it, and on each
-- axis thrust towards the speed that gets there and stops (at most `cruise`
-- m/s ahead, 1.5 m/s up or down): W/S ahead, A/D across, Space/Shift up and
-- down. In gravity letting go of Space is the way down, and Space comes
-- first: with it held only the one horizontal key that matters most is (with
-- a third key the lift engines could not hold the ship up).
-- Once within `tolerance` and slow, let go: the brake assist stops it.
local function fly_to(target, tolerance, timeout, cruise)
  cruise = cruise or 3
  local started = client.time()
  while true do
    local s = state()
    local p, v = s.scene_position, s.velocity
    local dx, dy, dz = target[1] - p[1], target[2] - p[2], target[3] - p[3]
    local flat = math.sqrt(dx * dx + dz * dz)
    if flat > 0.4 then
      face(dx, dz)
    end
    local yaw = math.rad(state().yaw_deg)
    local ahead = -math.sin(yaw) * v[1] - math.cos(yaw) * v[3]
    local across = math.cos(yaw) * v[1] - math.sin(yaw) * v[3]
    local towards = math.min(cruise, 0.7 * math.sqrt(4 * math.max(flat - 0.1, 0)))
    local rise = math.max(-1.5, math.min(1.5, 1.5 * dy))
    local heavy = s.gravity > 0.05
    if heavy and v[2] < rise then
      hold("shift", false)
      hold("space", true)
      if math.abs(ahead - towards) >= math.abs(across) then
        hold("a", false)
        hold("d", false)
        steer(ahead, towards, "w", "s", 0.25)
      else
        hold("w", false)
        hold("s", false)
        steer(across, 0, "d", "a", 0.25)
      end
    else
      if heavy then
        hold("space", false)
        hold("shift", false)
      else
        steer(v[2], rise, "space", "shift", 0.2)
      end
      steer(ahead, towards, "w", "s", 0.25)
      steer(across, 0, "d", "a", 0.25)
    end
    if flat <= tolerance and math.abs(dy) <= tolerance + 0.3 and s.speed < 0.6 then
      let_go()
      return s
    end
    if client.time() - started > timeout then
      let_go()
      error(string.format("could not reach %s in %d s: at %s, %.1f m/s", fmt(target), timeout, fmt(p), s.speed))
    end
    client.wait(0.05)
  end
end

-- 1. Online, in the demo area, with its bodies drawn.
client.wait_frames(1)
client.wait_until(function() return state().screen ~= "connecting" end, 20)
if state().screen ~= "online" then
  client.connect(state().address, 20)
end
client.wait_until(function()
  local s = state()
  assert(s.screen ~= "error", "cannot connect: " .. tostring(s.error))
  return s.screen == "online" and s.scene_position ~= nil
end, 20)
assert(state().scene == "demo-area", "the shard shows " .. tostring(state().scene) .. ", not the demo area")
client.wait_until(function() return #state().bodies >= 10 end, 10)
local s = state()
print(string.format("online: %d bodies, %d in view, gravity %.2f at %s", #s.bodies, s.bodies_in_view, s.gravity, fmt(s.scene_position)))
if distance(s.scene_position, { 0, 0, 0 }) < 3 then
  assert(s.gravity < 0.05, "the spawn point is not weightless")
end
client.screenshot("physics-1-spawn.png")

-- 2. The gravity room, as a newcomer sees it: turn right.
face(1, 0)
client.wait_frames(3)
client.screenshot("physics-2-gravity-room.png")

-- 3. Walk in at floor level: the ship falls the last centimetres and stands.
fly_to({ 12.5, 0.0, 1.5 }, 0.4, 40)
client.wait_until(function()
  local s = state()
  return s.landed and s.gravity > 9.5
end, 8)
s = state()
local floor_y = s.scene_position[2]
print(string.format("landed at %s, gravity %.2f m/s2", fmt(s.scene_position), s.gravity))
-- Look into the room (the chase camera behind, over the curb).
face(1, -0.35)
client.wait_frames(3)
client.screenshot("physics-3-landed.png")

-- 4. Lift off with the lift engines: Space for a second climbs.
client.key_down("space")
client.wait(0.6)
client.screenshot("physics-4-lift-off.png")
client.wait(0.4)
client.key_up("space")
s = state()
local climbed = s.scene_position[2] - floor_y
print(string.format("lifted off %.2f m", climbed))
assert(climbed > 1.0, string.format("Space for a second lifted the ship only %.2f m", climbed))
assert(not s.landed, "still landed after lifting off")

-- 5. A spring pendulum: fly into it, and it swings.
local bob = find("pendulum")
assert(bob, "no pendulum drawn")
local rest = bob.position
print("pendulum at " .. fmt(rest))
fly_to({ rest[1], rest[2] + 0.6, rest[3] - 2.2 }, 0.35, 40, 2)
fly_to({ rest[1], rest[2] + 0.6, rest[3] + 1.2 }, 0.5, 20, 2)
local swing = 0
local started = client.time()
local shot = false
while client.time() - started < 3 do
  local b = body(bob.guid)
  if b then
    local d = distance(b.position, rest)
    if d > swing then
      swing = d
    end
    if not shot and d > 0.25 then
      client.screenshot("physics-5-pendulum-swing.png")
      shot = true
    end
  end
  client.wait(0.05)
end
print(string.format("the pendulum swung %.2f m out", swing))
assert(swing > 0.25, string.format("the pendulum moved only %.2f m", swing))

-- 6. Out of the room again: weightless outside.
fly_to({ 8.0, 0.5, 1.5 }, 0.5, 40)
client.wait_until(function() return state().gravity < 0.05 end, 5)
print("left the gravity room at " .. fmt(at()))

-- 7. The zero-g corner: a floating crate drifts off when the ship flies into it.
local crate = find("crate", function(b) return b.position[1] < -10.5 and b.speed < 0.3 end)
assert(crate, "no floating crate in the zero-g corner")
local before = crate.position
print("floating crate at " .. fmt(before))
fly_to({ before[1] + 2.5, before[2] + 0.6, before[3] }, 0.35, 60)
fly_to({ before[1] - 1.0, before[2] + 0.6, before[3] }, 0.5, 20, 1.5)
client.wait(1.0)
local after = body(crate.guid)
assert(after, "the crate is no longer drawn")
local moved = distance(after.position, before)
print(string.format("the crate moved %.2f m and drifts at %.2f m/s", moved, after.speed))
assert(moved > 0.5, string.format("the crate moved only %.2f m", moved))
client.screenshot("physics-6-crate-pushed.png")

-- 8. The door: push its free end and it swings on its hinge.
local door = find("door")
assert(door, "no door drawn")
local closed = door.position
-- Round the east side of the corner (clear of the spring pair and the
-- door's post), then line up north of the door's free end.
fly_to({ -10.0, 0.0, -0.5 }, 0.5, 40)
fly_to({ -10.0, 0.0, closed[3] + 2.5 }, 0.5, 40)
fly_to({ closed[1] - 0.3, 0.0, closed[3] + 2.5 }, 0.35, 40)
fly_to({ closed[1] - 0.3, 0.0, closed[3] - 1.0 }, 0.5, 20, 1.5)
client.screenshot("physics-7-door-pushed.png")
client.wait(0.8)
local opened = distance(body(door.guid).position, closed)
print(string.format("the door's middle moved %.2f m", opened))
assert(opened > 0.15, string.format("the door moved only %.2f m", opened))
-- Step back (east, clear of the spring pair) and look at it: swung open
-- on its hinge.
fly_to({ closed[1] + 1.8, 0.0, closed[3] - 2.8 }, 0.5, 30)
local here = at()
face(closed[1] - here[1], closed[3] - here[3])
client.wait_frames(3)
client.screenshot("physics-8-door-open.png")

-- 9. The planetoid: up over the floor's corner, onto its top, and off again.
local planetoid = { -27.0, 10.0, -46.0 }
local radius = 4.0
fly_to({ -10.0, 18.0, -0.5 }, 0.6, 60, 4)
fly_to({ planetoid[1], 19.0, planetoid[3] }, 0.5, 90, 4)
s = state()
print(string.format("over the planetoid at %s, gravity %.2f m/s2", fmt(s.scene_position), s.gravity))
assert(s.gravity > 0.3, "no pull over the planetoid")
-- Down onto the top: the eye 1.7 m over the rock.
fly_to({ planetoid[1], planetoid[2] + radius + 1.75, planetoid[3] }, 0.3, 60, 2)
client.wait_until(function()
  local s = state()
  return s.landed and s.speed < 0.2
end, 15)
s = state()
local top = s.scene_position[2]
print(string.format("landed on the planetoid at %s, gravity %.2f m/s2", fmt(s.scene_position), s.gravity))
assert(math.abs(top - (planetoid[2] + radius + 1.7)) < 0.1, "not standing on the top")
face(27, 46)
client.wait_frames(3)
client.screenshot("physics-9-planetoid-landed.png")
client.hold("space", 1.0)
s = state()
local rose = s.scene_position[2] - top
print(string.format("lifted off the planetoid %.2f m", rose))
assert(rose > 1.0 and not s.landed, string.format("Space lifted the ship only %.2f m", rose))
-- Its moons keep circling: watch one for a few seconds from up here.
local moon = nil
for _, b in ipairs(state().bodies) do
  if b.visual == "ball" and distance(b.position, planetoid) < 11 and distance(b.position, planetoid) > 6 then
    moon = b
    break
  end
end
assert(moon, "no ball in orbit round the planetoid")
fly_to({ planetoid[1] + 14.0, planetoid[2] + 6.0, planetoid[3] + 14.0 }, 0.6, 60, 3)
face(-14, -14)
client.turn(0, -20)
local orbit = {}
for i = 1, 4 do
  client.wait_frames(2)
  client.screenshot(string.format("physics-10-orbit-%d.png", i))
  local b = body(moon.guid)
  assert(b, "the moon is no longer drawn")
  table.insert(orbit, { r = distance(b.position, planetoid), p = b.position })
  client.wait(1.2)
end
local travelled = distance(orbit[1].p, orbit[4].p)
for i, o in ipairs(orbit) do
  print(string.format("moon %d: %.2f m from the centre at %s", i, o.r, fmt(o.p)))
  assert(math.abs(o.r - orbit[1].r) < 0.3, "the moon left its orbit")
end
assert(travelled > 2.0, string.format("the moon moved only %.2f m", travelled))
client.turn(0, 20)

s = state()
print(string.format("prediction: last correction %.3f m, largest %.3f m in 5 s", s.correction_m, s.correction_max_m))
print("physics tour passed")
