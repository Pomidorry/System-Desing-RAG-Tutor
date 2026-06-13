# Load Balancing

A load balancer distributes incoming network traffic across multiple backend servers.

## Why Use It
- Prevents any single server from becoming a bottleneck
- Enables horizontal scaling
- Provides failover if a server goes down

## Algorithms
- **Round Robin** — requests rotate through servers in sequence
- **Least Connections** — route to the server with fewest active connections
- **IP Hash** — same client always hits the same server (useful for sticky sessions)

## Layer 4 vs Layer 7
- L4 (Transport): routes based on IP + TCP/UDP port
- L7 (Application): routes based on HTTP headers, URL path, cookies
