# bug1_full_fixed.py
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

# ---------------- File I/O ----------------
def read_polygons(filename):
    polys = []
    with open(filename, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]
    i = 0
    while i < len(lines):
        try:
            n = int(lines[i])
        except Exception:
            i += 1
            continue
        coords = []
        for k in range(n):
            i += 1
            x_str, y_str = lines[i].split()
            coords.append((float(x_str), float(y_str)))
        polys.append(Polygon(coords))
        i += 1
    return polys

# ---------------- Inflation / validation ----------------
def inflate_obstacles(obstacles, robot_radius):
    if robot_radius <= 0:
        return obstacles
    return [obs.buffer(robot_radius, resolution=16) for obs in obstacles]

def validate_start_goal_against_inflated(start_pt, goal_pt, inflated_obstacles, workspace):
    # Ensure start and goal are inside workspace and not inside any inflated obstacle
    if not workspace.contains(start_pt):
        raise ValueError("Start not inside safe workspace.")
    if not workspace.contains(goal_pt):
        raise ValueError("Goal not inside safe workspace.")
    for obs in inflated_obstacles:
        if obs.contains(start_pt) or obs.touches(start_pt):
            raise ValueError("Start is inside or too close to an inflated obstacle.")
        if obs.contains(goal_pt) or obs.touches(goal_pt):
            raise ValueError("Goal is inside or too close to an inflated obstacle.")

# ---------------- Merge touching inflated obstacles into workspace holes ----------------
def merge_obstacles_with_boundary(boundary, inflated_obstacles):
    """
    boundary: the deflated (safe) boundary polygon (or original if robot radius 0)
    inflated_obstacles: list of Polygons (inflated)
    Returns (workspace_polygon, effective_obstacles_list)
      - workspace_polygon: boundary minus merged_touching obstacles (largest polygon chosen if MultiPolygon)
      - effective_obstacles_list: list of obstacles that are not touching boundary + holes (Polygons) created from touched obstacles
    """
    touching = []
    non_touching = []
    for obs in inflated_obstacles:
        if obs.intersects(boundary) or obs.touches(boundary):
            touching.append(obs)
        else:
            non_touching.append(obs)

    holes_as_obstacles = []
    if touching:
        merged_touching = unary_union(touching)
        boundary_with_holes = boundary.difference(merged_touching)

        # If MultiPolygon choose largest one as workspace (the usable interior)
        if boundary_with_holes.geom_type == "MultiPolygon":
            workspace = max(boundary_with_holes.geoms, key=lambda p: p.area)
        else:
            workspace = boundary_with_holes

        # Each interior ring of workspace corresponds to a hole; convert to Polygon obstacles
        for interior in workspace.interiors:
            holes_as_obstacles.append(Polygon(interior))
    else:
        workspace = boundary

    effective_obstacles = non_touching + holes_as_obstacles
    return workspace, effective_obstacles

# ---------------- Random start/goal (use workspace & inflated obstacles) ----------------
def random_point_outside_obstacles(workspace, inflated_obstacles, max_tries=2000):
    
    minx, miny, maxx, maxy = workspace.bounds
    for _ in range(max_tries):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if not workspace.contains(p):
            continue
        bad = False
        for obs in inflated_obstacles:
            if obs.contains(p) or obs.touches(p):
                bad = True
                break
        if bad:
            continue
        return (x, y)
    raise RuntimeError("Could not find a valid random point outside obstacles.")

def generate_random_start_goal(workspace, inflated_obstacles):
    start = random_point_outside_obstacles(workspace, inflated_obstacles)
    goal = random_point_outside_obstacles(workspace, inflated_obstacles)
    tries = 0
    while Point(start).distance(Point(goal)) < 1.0 and tries < 500:
        goal = random_point_outside_obstacles(workspace, inflated_obstacles)
        tries += 1
    return start, goal

# ---------------- Intersection helpers ----------------
def find_first_intersection_along_line(p_from, p_to, obstacles, workspace=None):
    motion = LineString([p_from, p_to])
    best_pt, best_obs, best_dist = None, None, float('inf')

    # workspace exterior
    if workspace is not None and hasattr(workspace, "exterior"):
        inter = motion.intersection(workspace.exterior)
        if not inter.is_empty:
            if inter.geom_type == "Point":
                candidates = [inter]
            elif inter.geom_type == "MultiPoint":
                candidates = list(inter.geoms)
            elif inter.geom_type == "LineString":
                coords = list(inter.coords)
                candidates = [Point(coords[0]), Point(coords[-1])]
            else:
                candidates = []
            for p in candidates:
                d = p_from.distance(p)
                if d < best_dist:
                    best_dist, best_pt, best_obs = d, p, workspace

    # obstacles
    for obs in obstacles:
        inter = motion.intersection(obs.boundary)
        if inter.is_empty:
            if motion.intersects(obs):
                # fallback: nearest boundary point
                p = obs.exterior.interpolate(obs.exterior.project(p_from))
                d = p_from.distance(p)
                if d < best_dist:
                    best_dist, best_pt, best_obs = d, p, obs
            continue

        if inter.geom_type == "Point":
            candidates = [inter]
        elif inter.geom_type == "MultiPoint":
            candidates = list(inter.geoms)
        elif inter.geom_type == "LineString":
            coords = list(inter.coords)
            candidates = [Point(coords[0]), Point(coords[-1])]
        else:
            candidates = []
        for p in candidates:
            d = p_from.distance(p)
            if d < best_dist:
                best_dist, best_pt, best_obs = d, p, obs

    # final fallback: if motion intersects interior of any obstacle but didn't hit boundary above
    if best_pt is None:
        for obs in obstacles:
            if motion.intersects(obs):
                p = obs.exterior.interpolate(obs.exterior.project(p_from))
                d = p_from.distance(p)
                if d < best_dist:
                    best_dist, best_pt, best_obs = d, p, obs
        if best_pt is None and workspace is not None and motion.intersects(workspace):
            p = workspace.exterior.interpolate(workspace.exterior.project(p_from))
            best_pt, best_obs = p, workspace

    return best_pt, best_obs

# ---------------- Boundary sampling ----------------
def sample_full_boundary_from(obstacle, start_point, step_size, goal_point):
    """
    Sample a full loop of points along the polygon piece nearest start_point (handles MultiPolygon).
    Returns (circ_points_list, leave_point, leave_idx).
    """
    poly_piece = None
    if obstacle.geom_type == "Polygon":
        poly_piece = obstacle
    else:
        best_d = float('inf')
        for g in obstacle.geoms:
            d = g.exterior.distance(start_point)
            if d < best_d:
                best_d = d
                poly_piece = g
    if poly_piece is None:
        raise RuntimeError("Unable to select polygon piece for sampling.")

    boundary = poly_piece.exterior
    L = boundary.length
    s0 = boundary.project(start_point)
    start_proj = boundary.interpolate(s0)

    n_steps = max(8, int(math.ceil(L / step_size)))
    circ_points = [start_proj]
    best_d = start_proj.distance(goal_point)
    leave_pt, leave_idx = start_proj, 0

    for k in range(1, n_steps + 1):
        t_mod = (s0 + k * step_size) % L
        p = boundary.interpolate(t_mod)
        circ_points.append(p)
        d = p.distance(goal_point)
        if d < best_d - 1e-12:
            best_d, leave_pt, leave_idx = d, p, len(circ_points) - 1

    if circ_points[-1].distance(start_proj) > 1e-6:
        circ_points.append(start_proj)

    return circ_points, leave_pt, leave_idx

# ---------------- Bug1 (full loop then leave at best point) ----------------
def bug1_discrete(start, goal, obstacles, workspace,
                  step_size=0.25, boundary_step=0.25, goal_tol=0.5, max_iters=20000):
    start_pt, goal_pt = Point(start), Point(goal)
    path = [start_pt]
    current = start_pt
    iters = 0
    last_hit_obs = None

    if not (hasattr(workspace, "contains") and workspace.contains(start_pt) and workspace.contains(goal_pt)):
        raise ValueError("Start or goal not inside workspace (after inflation/merging).")

    while current.distance(goal_pt) > goal_tol and iters < max_iters:
        iters += 1
        vec = np.array([goal_pt.x - current.x, goal_pt.y - current.y])
        dist_to_goal = np.linalg.norm(vec)
        if dist_to_goal < 1e-9:
            break
        dir_unit = vec / dist_to_goal
        next_pt = Point((current.x + dir_unit[0] * step_size,
                         current.y + dir_unit[1] * step_size))
        seg = LineString([current, next_pt])

        collision = (not workspace.contains(next_pt)) or any(seg.intersects(obs) for obs in obstacles)
        if not collision:
            path.append(next_pt)
            current = next_pt
            continue

        hit_pt, hit_obs = find_first_intersection_along_line(current, goal_pt, obstacles, workspace)
        if hit_pt is None or hit_obs is None:
            print("⚠️ Collision but no intersection found; stopping.")
            break

        # Avoid oscillating on same object repeatedly: allow one bypass attempt
        if last_hit_obs is not None and hasattr(hit_obs, "equals") and hit_obs.equals(last_hit_obs):
            # attempt a larger nudge forward; if not possible, break
            vec2 = np.array([goal_pt.x - current.x, goal_pt.y - current.y])
            d2 = np.linalg.norm(vec2)
            if d2 > 1e-9:
                dir_unit2 = vec2 / d2
                off_pt = Point((current.x + dir_unit2[0] * step_size * 2,
                                current.y + dir_unit2[1] * step_size * 2))
                if workspace.contains(off_pt) and not any(off_pt.within(o) for o in obstacles):
                    path.append(off_pt)
                    current = off_pt
                    continue
            break

        circ_points, leave_pt, leave_idx = sample_full_boundary_from(hit_obs, hit_pt, boundary_step, goal_pt)
        last_hit_obs = hit_obs

        # walk full loop
        for p in circ_points:
            if path[-1].distance(p) > 1e-9:
                path.append(p)

        # then walk to leave point (closest to goal)
        if leave_idx > 0:
            for k in range(1, leave_idx + 1):
                if path[-1].distance(circ_points[k]) > 1e-9:
                    path.append(circ_points[k])
            current = circ_points[leave_idx]
        else:
            current = circ_points[0]
            if path[-1].distance(current) > 1e-9:
                path.append(current)

        # try to nudge off the boundary toward the goal
        vec2 = np.array([goal_pt.x - current.x, goal_pt.y - current.y])
        d2 = np.linalg.norm(vec2)
        if d2 > 1e-9:
            dir_unit2 = vec2 / d2
            for scale in [1.0, 1.5, 2.0]:
                off_pt = Point((current.x + dir_unit2[0] * step_size * scale,
                                current.y + dir_unit2[1] * step_size * scale))
                seg2 = LineString([current, off_pt])
                if workspace.contains(off_pt) and not any(seg2.intersects(obs) for obs in obstacles):
                    if path[-1].distance(off_pt) > 1e-9:
                        path.append(off_pt)
                    current = off_pt
                    break

    if iters >= max_iters:
        print("Max iterations reached; stopping.")
    return path

# ---------------- Visualization ----------------
def _plot_polygon_or_multipolygon(ax, geom, facecolor=None, edgecolor='k', lw=1.0, alpha=1.0, label=None, label_once=False):
    """Helper to plot a Polygon or MultiPolygon; label only first time if label_once True."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        if facecolor is not None:
            ax.fill(x, y, color=facecolor, alpha=alpha)
        ax.plot(x, y, color=edgecolor, lw=lw, label=label)
    else:
        # MultiPolygon or GeometryCollection: iterate
        first = True
        for g in geom.geoms:
            lab = label if (first and label is not None and not label_once) else None
            _plot_polygon_or_multipolygon(ax, g, facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha, label=lab)
            first = False

def animate_path(path_points, effective_obstacles, workspace, safe_boundary,
                 inflated_obstacles=None, original_obstacles=None,
                 start=None, goal=None, obstacle_file="out", interval=40):
    coords = [(p.x, p.y) for p in path_points]
    if len(coords) == 0:
        print("No path to animate.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Workspace (shows holes as interior rings)
    _plot_polygon_or_multipolygon(ax, workspace, facecolor=None, edgecolor='b', lw=2, alpha=1.0, label="Workspace")

    # Deflated safe boundary (outer safe limit) - might be MultiPolygon
    if safe_boundary is not None and not safe_boundary.is_empty:
        if safe_boundary.geom_type == "Polygon":
            x, y = safe_boundary.exterior.xy
            ax.plot(x, y, "r--", lw=1.5, label="Deflated Boundary")
        else:
            plotted = False
            for g in safe_boundary.geoms:
                x, y = g.exterior.xy
                if not plotted:
                    ax.plot(x, y, "r--", lw=1.5, label="Deflated Boundary")
                    plotted = True
                else:
                    ax.plot(x, y, "r--", lw=1.5)

    # Fill holes inside workspace (these are interior rings; they appear as 'obstacles')
    for hole in workspace.interiors:
        hx, hy = hole.xy
        ax.fill(hx, hy, color="lightgray", alpha=0.6)

    # Plot effective (inflated non-touching + holes_as_obstacles)
    for i, obs in enumerate(effective_obstacles):
        x, y = obs.exterior.xy
        ax.fill(x, y, color="gray", alpha=0.6)
        ax.plot(x, y, "k-", lw=1.2, alpha=0.9)

    # Plot inflated outlines (dashed)
    if inflated_obstacles:
        plotted = False
        for obs in inflated_obstacles:
            x, y = obs.exterior.xy
            if not plotted:
                ax.plot(x, y, "r--", lw=1.1, alpha=0.9, label="Inflated Obstacles")
                plotted = True
            else:
                ax.plot(x, y, "r--", lw=1.1, alpha=0.9)

    # Plot original obstacles (dotted outlines)
    if original_obstacles:
        plotted = False
        for obs in original_obstacles:
            x, y = obs.exterior.xy
            if not plotted:
                ax.plot(x, y, "m:", lw=1.0, alpha=0.9, label="Original Obstacles")
                plotted = True
            else:
                ax.plot(x, y, "m:", lw=1.0, alpha=0.9)

    # Start/Goal
    if start:
        ax.plot(start[0], start[1], "go", markersize=9, label="Start")
    if goal:
        ax.plot(goal[0], goal[1], "ro", markersize=9, label="Goal")

    path_line, = ax.plot([], [], "-r", lw=2, label="Path")
    robot_dot, = ax.plot([], [], "bo", markersize=6)

    ax.set_aspect("equal", "box")
        # Set axis limits with extra margin around boundary
    minx, miny, maxx, maxy = workspace.bounds
    margin_x = (maxx - minx) * 0.1  # 10% margin on each side
    margin_y = (maxy - miny) * 0.1

    ax.set_xlim(minx - margin_x, maxx + margin_x)
    ax.set_ylim(miny - margin_y, maxy + margin_y)

    # minx, miny, maxx, maxy = workspace.bounds
    # ax.set_xlim(minx, maxx)
    # ax.set_ylim(miny, maxy)
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.set_title("Bug1 Navigation (Deflated workspace and inflated obstacles)")

    def init():
        path_line.set_data([], [])
        robot_dot.set_data([], [])
        return path_line, robot_dot

    def update(frame):
        xs, ys = zip(*coords[:frame + 1])
        path_line.set_data(xs, ys)
        robot_dot.set_data([xs[-1]], [ys[-1]])
        return path_line, robot_dot

    anim = FuncAnimation(fig, update, frames=len(coords), init_func=init,
                         interval=interval, blit=True, repeat=False)

    gif_file = f"{os.path.splitext(os.path.basename(obstacle_file))[0]}.gif"
    try:
        anim.save(gif_file, writer=PillowWriter(fps=max(1, 1000 // interval)))
        print("✅ Saved animation as", gif_file)
    except Exception as e:
        print("Could not save gif:", e)
    plt.show()


def get_start_goal_by_click(workspace, inflated_obstacles, boundary_orig):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw boundary
    bx, by = boundary_orig.exterior.xy
    ax.plot(bx, by, "b-", lw=2, label="Boundary")

    # Draw inflated obstacles
    for obs in inflated_obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, color="gray", alpha=0.5)

    ax.set_title("Click START (green) then GOAL (red)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    points = []

    def onclick(event):
        if event.inaxes != ax:
            return
        p = Point(event.xdata, event.ydata)

        # Validate click
        if not workspace.contains(p):
            print("❌ Point outside workspace")
            return
        for obs in inflated_obstacles:
            if obs.contains(p) or obs.touches(p):
                print("❌ Point inside obstacle")
                return

        points.append((event.xdata, event.ydata))

        if len(points) == 1:
            ax.plot(event.xdata, event.ydata, "go", markersize=9)
            ax.set_title("Now click GOAL (red)")
        elif len(points) == 2:
            ax.plot(event.xdata, event.ydata, "ro", markersize=9)
            plt.close(fig)

        fig.canvas.draw()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(points) < 2:
        raise RuntimeError("Start/Goal not selected properly.")

    return points[0], points[1]

# ---------------- Main ----------------
if __name__ == "__main__":
    # --- params & input ---
    try:
        idx = int(input("Enter obstacle file number (1–8) [default 1]: ").strip()) - 1
        if idx < 0:
            idx = 0
    except Exception:
        idx = 0
    obstacle_file = f"obstacles{idx+1}.txt"
    boundary_file = "Bdry.txt"

    try:
        ROBOT_RADIUS = float(input("Enter robot radius (e.g. 0.5) [default 0]: ").strip())
        if ROBOT_RADIUS < 0:
            ROBOT_RADIUS = 0.0
    except Exception:
        ROBOT_RADIUS = 0.0

    # load
    obstacles_orig = read_polygons(obstacle_file)
    boundary_orig = read_polygons(boundary_file)[0]

    # inflate obstacles
    inflated = inflate_obstacles(obstacles_orig, ROBOT_RADIUS)

    # deflate boundary for robot center safety
    safe_boundary = boundary_orig.buffer(-ROBOT_RADIUS, resolution=16) if ROBOT_RADIUS > 0 else boundary_orig
    if safe_boundary.is_empty:
        print("⚠️ Deflated boundary is empty; falling back to original boundary.")
        safe_boundary = boundary_orig

    # merge inflated obstacles that touch boundary into holes of workspace
    workspace, effective_obstacles = merge_obstacles_with_boundary(safe_boundary, inflated)

    # get start/goal (random or manual)
    USE_RANDOM = False
    USE_MOUSE = True

    if USE_RANDOM:
        start, goal = generate_random_start_goal(workspace, inflated)
        print("Start:", start, "Goal:", goal)

    elif USE_MOUSE:
        start, goal = get_start_goal_by_click(workspace, inflated, boundary_orig)
        print("Start:", start, "Goal:", goal)

    else:
        sx, sy = map(float, input("Enter start x y: ").split())
        gx, gy = map(float, input("Enter goal x y: ").split())
        start, goal = (sx, sy), (gx, gy)

        validate_start_goal_against_inflated(Point(start), Point(goal), inflated, workspace)

    # final sanity
    try:
        validate_start_goal_against_inflated(Point(start), Point(goal), inflated, workspace)
    except Exception as e:
        print("Start/goal validation failed:", e)
        print("Visualizing setup to diagnose...")
        # quick visualization then exit
        fig, ax = plt.subplots(figsize=(7,7))
        # draw original boundary
        bx, by = boundary_orig.exterior.xy
        ax.plot(bx, by, "b-")
        # draw inflated obstacles
        for obs in inflated:
            x,y = obs.exterior.xy
            ax.plot(x,y,"r--")
        ax.plot(start[0], start[1], "go")
        ax.plot(goal[0], goal[1], "ro")
        ax.set_title("Validation view")
        plt.show()
        raise

    # plan
    path_pts = bug1_discrete(start, goal, effective_obstacles, workspace,
                             step_size=0.25, boundary_step=0.25, goal_tol=0.5, max_iters=20000)
    print(f"[{obstacle_file}] Path length:", len(path_pts))

    # animate (pass safe_boundary and inflated and original obstacles for visualization)
    animate_path(path_pts,
                 effective_obstacles,
                 workspace,
                 safe_boundary,
                 inflated_obstacles=inflated,
                 original_obstacles=obstacles_orig,
                 start=start,
                 goal=goal,
                 obstacle_file=obstacle_file,
                 interval=40)
