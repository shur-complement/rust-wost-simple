//! Rust port of the "Walk on Stars" algorithm for Laplace equations.
//! Based on https://github.com/GeometryCollective/wost-simple

use rand::{thread_rng, Rng};
use std::f64::consts::PI;
use std::f64::INFINITY;
use std::io::Write;

/// returns a random value in the range [min,max]
fn random(min: f64, max: f64) -> f64 {
  thread_rng().gen_range(min..=max)
}

// use num_complex to implement 2D vectors
type Vec2D = num_complex::Complex64;

fn vec2d(re: f64, im: f64) -> Vec2D {
  Vec2D::new(re, im)
}

fn length(u: &Vec2D) -> f64 {
  u.norm()
}

fn angle_of(u: &Vec2D) -> f64 {
  u.arg()
}

fn rot90(u: &Vec2D) -> Vec2D {
  Vec2D::new(-u.im, u.re)
}

fn dot(u: &Vec2D, v: &Vec2D) -> f64 {
  u.re * v.re + u.im * v.im
}

fn cross(u: &Vec2D, v: &Vec2D) -> f64 {
  u.re * v.im - u.im * v.re
}

fn clamp(val: f64, lo: f64, hi: f64) -> f64 {
  val.max(lo).min(hi)
}

/// returns the closest point to x on a segment with endpoints a and b
fn closest_point(x: &Vec2D, a: &Vec2D, b: &Vec2D) -> Vec2D {
  let u = b - a;
  let t = clamp(dot(&(x - a), &u) / dot(&u, &u), 0.0, 1.0);
  (1.0 - t) * a + t * b
}

/// returns true if the point b on the polyline abc is a silhoutte relative to x
fn is_silhouette(x: &Vec2D, a: &Vec2D, b: &Vec2D, c: &Vec2D) -> bool {
  cross(&(b - a), &(x - a)) * cross(&(c - b), &(x - b)) < 0.0
}

/// returns the time t at which the ray x+tv intersects segment ab,
/// or infinity if there is no intersection
fn ray_intersection(x: &Vec2D, v: &Vec2D, a: &Vec2D, b: &Vec2D) -> f64 {
  let u = b - a;
  let w = x - a;
  let d = cross(v, &u);
  let s = cross(v, &w) / d;
  let t = cross(&u, &w) / d;
  if t > 0. && 0. <= s && s <= 1. {
    return t;
  }
  INFINITY
}

// boundary geometry is represented by polylines
type Polyline = Vec<Vec2D>;

/// returns distance from x to closest point on the given polylines P
fn distance_polylines(x: &Vec2D, P: &Vec<Polyline>) -> f64 {
  let mut d = INFINITY; // minimum distance so far
  for i in 0..P.len() {
    for j in 0..(P[i].len() - 1) {
      let y = closest_point(x, &P[i][j], &P[i][j + 1]); // distance to segment
      d = d.min(length(&(x - y))); // update minimum distance
    }
  }
  d
}

/// returns distance from x to closest silhouette point on the given polylines P
fn silhouette_distance_polylines(x: &Vec2D, P: &Vec<Polyline>) -> f64 {
  let mut d = INFINITY; // minimum distance so far
  for i in 0..P.len() {
    for j in 1..(P[i].len() - 1) {
      if is_silhouette(x, &P[i][j - 1], &P[i][j], &P[i][j + 1]) {
        d = d.min(length(&(x - P[i][j]))); // update minimum distance
      }
    }
  }
  d
}

/// finds the first intersection y of the ray x+tv with the given polylines P,
/// restricted to a ball of radius r around x.  The flag onBoundary indicates
/// whether the first hit is on a boundary segment (rather than the sphere), and
/// if so sets n to the normal at the hit point.
fn intersect_polylines(
  x: &Vec2D,
  v: &Vec2D,
  r: f64,
  P: &Vec<Polyline>,
  n: &mut Vec2D,
  on_boundary: &mut bool,
) -> Vec2D {
  let mut t_min = r; // smallest hit time so far
  *n = Vec2D::new(0.0, 0.0); // first hit normal
  *on_boundary = false; // will be true only if the first hit is on a segment
  for i in 0..P.len() {
    for j in 0..(P[i].len() - 1) {
      let c = 1e-5; // ray offset (to avoid self-intersection)
      let t = ray_intersection(&(x + c * v), v, &P[i][j], &P[i][j + 1]);
      if t < t_min { // closest hit so far
        t_min = t;
        *n = rot90(&(P[i][j + 1] - P[i][j])); // get normal
        *n /= length(n); // make normal unit length
        *on_boundary = true;
      }
    }
  }
  x + t_min * v // first hit location
}

/// solves a Laplace equation Delta u = 0 at x0, where the Dirichlet and Neumann
/// boundaries are each given by a collection of polylines, the Neumann
/// boundary conditions are all zero, and the Dirichlet boundary conditions
/// are given by a function g that can be evaluated at any point in space
fn solve(
  x0: &Vec2D,                         // evaluation point
  boundary_dirichlet: &Vec<Polyline>, // absorbing part of the boundary
  boundary_neumann: &Vec<Polyline>,   // reflecting part of the boundary
  g: impl Fn(&Vec2D) -> f64,          // Dirichlet boundary values
) -> f64 {
  let eps = 0.0001f64;   // stopping tolerance
  let r_min = 0.0001f64; // minimum step size
  let n_walks = 65536;   // number of Monte Carlo samples
  let max_steps = 65536; // maximum walk length

  let mut sum = 0.0; // running sum of boundary contributions
  for _ in 0..n_walks {
    let mut x = *x0; // start walk at the evaluation point
    let mut n = vec2d(0.0, 0.0); // assume x0 is an interior point, and has no normal
    let mut on_boundary = false; // flag whether x is on the interior or boundary

    let mut r;
    let mut d_dirichlet;
    let mut d_silhouette; // radii used to define star shaped region
    let mut steps = 0;
    loop {
      // loop until the walk hits the Dirichlet boundary
      // compute the radius of the largest star-shaped region
      d_dirichlet = distance_polylines(&x, boundary_dirichlet);
      d_silhouette = silhouette_distance_polylines(&x, boundary_neumann);
      r = r_min.max(d_dirichlet.min(d_silhouette));

      // intersect a ray with the star-shaped region boundary
      let mut theta = random(-PI, PI);
      if on_boundary { // sample from a hemisphere around the normal
        theta = theta / 2. + angle_of(&n);
      }
      let v = vec2d(theta.cos(), theta.sin()); // unit ray direction
      x = intersect_polylines(&x, &v, r, &boundary_neumann, &mut n, &mut on_boundary);

      steps += 1;

      //stop if we hit the Dirichlet boundary, or the walk is too long
      if !(d_dirichlet > eps && steps < max_steps) {
        break;
      }
    }

    if steps >= max_steps {
      eprintln!("Hit max steps");
    }

    sum += g(&x); // accumulate contribution of the boundary value
  }
  sum / (n_walks as f64) // Monte Carlo estimate
}

fn lines(x: &Vec2D) -> f64 {
  let s = 8.0;
  (s * x.re).floor() % 2.0
}

/// checks whether a given evaluation point is actually inside the domain
fn signed_angle(x: &Vec2D, P: &Vec<Polyline>) -> f64 {
  let mut theta = 0.;
  for i in 0..P.len() {
    for j in 0..(P[i].len() - 1) {
      theta += angle_of(&((P[i][j + 1] - x) / (P[i][j] - x)));
    }
  }
  theta
}

/// Returns true if the point x is contained in the region bounded by the Dirichlet
/// and Neumann curves.  We assume these curves form a collection of closed polygons,
/// and are given in a consistent counter-clockwise winding order.
fn inside_domain(
  x: &Vec2D,
  boundary_dirichlet: &Vec<Polyline>,
  boundary_neumann: &Vec<Polyline>,
) -> bool {
  let theta = signed_angle(x, boundary_dirichlet) + signed_angle(x, boundary_neumann);
  let delta = 1e-4; // numerical tolerance
  (theta - 2. * PI).abs() < delta // boundary winds around x exactly once
}

fn main() -> std::io::Result<()> {
  
  // for simplicity, in this code we assume that the Dirichlet and Neumann
  // boundary polylines form a collection of closed polygons (possibly with holes),
  // and are given with consistent counter-clockwise orientation
  let boundary_dirichlet = vec![
    vec![vec2d(0.2, 0.2), vec2d(0.6, 0.0), vec2d(1.0, 0.2)],
    vec![vec2d(1.0, 1.0), vec2d(0.6, 0.8), vec2d(0.2, 1.0)],
  ];

  let boundary_neumann = vec![
    vec![vec2d(1.0, 0.2), vec2d(0.8, 0.6), vec2d(1.0, 1.0)],
    vec![vec2d(0.2, 1.0), vec2d(0.0, 0.6), vec2d(0.2, 0.2)],
  ];

  let file = std::fs::File::create("out.csv")?;
  let mut out = std::io::BufWriter::new(file);

  let s = 128; // image size
  for j in 0..s {
    eprintln!("row {} of {}", j, s);
    for i in 0..s {
      let x0 = vec2d(
        ((i as f64) + 0.5) / (s as f64),
        ((j as f64) + 0.5) / (s as f64),
      );
      let mut u = 0.;
      if inside_domain(&x0, &boundary_dirichlet, &boundary_neumann) {
        u = solve(&x0, &boundary_dirichlet, &boundary_neumann, lines);
      }
      write!(out, "{}", u)?;
      if i < s - 1 {
        write!(out, ",")?;
      }
    }
    writeln!(out, "");
    out.flush()?;
  }
  out.flush()?;
  Ok(())
}
