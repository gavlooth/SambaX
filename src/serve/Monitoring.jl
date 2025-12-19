module Monitoring

"""
Production Monitoring for NER Service.

Provides:
- Prometheus-compatible metrics
- Request latency tracking
- Entity extraction statistics
- Error rate monitoring
- Alert threshold checking

Metrics exposed:
- ner_requests_total: Counter of total requests
- ner_request_latency_seconds: Histogram of request latencies
- ner_entities_total: Counter of entities by type
- ner_batch_size: Histogram of batch sizes
- ner_errors_total: Counter of errors by type
"""

using Dates
using Statistics

# =============================================================================
# Metric Types
# =============================================================================

"""
Counter: Monotonically increasing value.
"""
mutable struct Counter
    name::String
    help::String
    value::Float64
    labels::Dict{String, Float64}  # label_value => count
end

function Counter(name::String, help::String; labels::Vector{String} = String[])
    label_values = Dict{String, Float64}()
    for label in labels
        label_values[label] = 0.0
    end
    return Counter(name, help, 0.0, label_values)
end

function inc!(c::Counter, value::Float64 = 1.0)
    c.value += value
end

function inc!(c::Counter, label::String, value::Float64 = 1.0)
    c.labels[label] = get(c.labels, label, 0.0) + value
end

"""
Histogram: Distribution of values with configurable buckets.
"""
mutable struct Histogram
    name::String
    help::String
    buckets::Vector{Float64}
    bucket_counts::Vector{Int}
    sum::Float64
    count::Int
end

function Histogram(name::String, help::String; buckets::Vector{Float64} = default_buckets())
    sorted_buckets = sort(buckets)
    return Histogram(
        name, help, sorted_buckets,
        zeros(Int, length(sorted_buckets) + 1),  # +1 for +Inf
        0.0, 0
    )
end

function default_buckets()
    return [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
end

function observe!(h::Histogram, value::Float64)
    h.sum += value
    h.count += 1

    # Increment bucket counts
    for (i, bucket) in enumerate(h.buckets)
        if value <= bucket
            h.bucket_counts[i] += 1
        end
    end
    # +Inf bucket always incremented
    h.bucket_counts[end] += 1
end

"""
Gauge: Value that can go up and down.
"""
mutable struct Gauge
    name::String
    help::String
    value::Float64
end

function Gauge(name::String, help::String)
    return Gauge(name, help, 0.0)
end

function set!(g::Gauge, value::Float64)
    g.value = value
end

function inc!(g::Gauge, value::Float64 = 1.0)
    g.value += value
end

function dec!(g::Gauge, value::Float64 = 1.0)
    g.value -= value
end

# =============================================================================
# NER Metrics Registry
# =============================================================================

"""
Registry of all NER service metrics.
"""
mutable struct MetricsRegistry
    # Request metrics
    requests_total::Counter
    request_latency::Histogram
    errors_total::Counter

    # NER-specific metrics
    entities_total::Counter  # By entity type
    tokens_processed::Counter
    batch_size::Histogram

    # System metrics
    active_requests::Gauge
    model_load_time::Gauge

    # Tracking
    last_reset::DateTime
end

function MetricsRegistry()
    entity_types = [
        "PERSON", "AGENCY", "PLACE", "ORGANISM", "EVENT",
        "INSTRUMENT", "WORK", "DOMAIN", "MEASURE"
    ]

    return MetricsRegistry(
        Counter("ner_requests_total", "Total NER requests"),
        Histogram("ner_request_latency_seconds", "Request latency in seconds",
            buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        Counter("ner_errors_total", "Total errors by type"),

        Counter("ner_entities_total", "Entities extracted by type",
            labels = entity_types),
        Counter("ner_tokens_processed", "Total tokens processed"),
        Histogram("ner_batch_size", "Batch sizes",
            buckets = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),

        Gauge("ner_active_requests", "Currently active requests"),
        Gauge("ner_model_load_time_seconds", "Model load time"),

        now()
    )
end

# Global registry
const METRICS = Ref{Union{MetricsRegistry, Nothing}}(nothing)

function get_metrics()
    if METRICS[] === nothing
        METRICS[] = MetricsRegistry()
    end
    return METRICS[]
end

function reset_metrics!()
    METRICS[] = MetricsRegistry()
end

# =============================================================================
# Recording Functions
# =============================================================================

"""
Record a completed request.
"""
function record_request!(
    latency::Float64,
    batch_size::Int = 1,
    entities::Vector = [],
    num_tokens::Int = 0
)
    m = get_metrics()

    inc!(m.requests_total)
    observe!(m.request_latency, latency)
    observe!(m.batch_size, Float64(batch_size))

    if num_tokens > 0
        inc!(m.tokens_processed, Float64(num_tokens))
    end

    # Count entities by type
    for entity in entities
        entity_type = if hasfield(typeof(entity), :label)
            entity.label
        elseif entity isa NamedTuple && haskey(entity, :label)
            entity.label
        elseif entity isa Tuple && length(entity) >= 2
            entity[2]  # Assume (text, type, ...) format
        else
            "UNKNOWN"
        end
        inc!(m.entities_total, entity_type)
    end
end

"""
Record an error.
"""
function record_error!(error_type::String = "unknown")
    m = get_metrics()
    inc!(m.errors_total, error_type)
end

"""
Track active requests (for concurrency monitoring).
"""
function track_active_request(f::Function)
    m = get_metrics()
    inc!(m.active_requests)
    try
        return f()
    finally
        dec!(m.active_requests)
    end
end

"""
Record model load time.
"""
function record_model_load_time!(seconds::Float64)
    m = get_metrics()
    set!(m.model_load_time, seconds)
end

# =============================================================================
# Prometheus Export
# =============================================================================

"""
Format metrics in Prometheus text format.
"""
function prometheus_format()
    m = get_metrics()
    lines = String[]

    # Helper to format a metric
    function format_counter(c::Counter)
        push!(lines, "# HELP $(c.name) $(c.help)")
        push!(lines, "# TYPE $(c.name) counter")

        if isempty(c.labels)
            push!(lines, "$(c.name) $(c.value)")
        else
            # Total without labels
            push!(lines, "$(c.name) $(c.value)")
            # Per-label values
            for (label, value) in c.labels
                if value > 0
                    push!(lines, "$(c.name){type=\"$label\"} $value")
                end
            end
        end
    end

    function format_histogram(h::Histogram)
        push!(lines, "# HELP $(h.name) $(h.help)")
        push!(lines, "# TYPE $(h.name) histogram")

        cumulative = 0
        for (i, bucket) in enumerate(h.buckets)
            cumulative += h.bucket_counts[i]
            push!(lines, "$(h.name)_bucket{le=\"$bucket\"} $cumulative")
        end
        push!(lines, "$(h.name)_bucket{le=\"+Inf\"} $(h.count)")
        push!(lines, "$(h.name)_sum $(h.sum)")
        push!(lines, "$(h.name)_count $(h.count)")
    end

    function format_gauge(g::Gauge)
        push!(lines, "# HELP $(g.name) $(g.help)")
        push!(lines, "# TYPE $(g.name) gauge")
        push!(lines, "$(g.name) $(g.value)")
    end

    # Format all metrics
    format_counter(m.requests_total)
    push!(lines, "")

    format_histogram(m.request_latency)
    push!(lines, "")

    format_counter(m.errors_total)
    push!(lines, "")

    format_counter(m.entities_total)
    push!(lines, "")

    format_counter(m.tokens_processed)
    push!(lines, "")

    format_histogram(m.batch_size)
    push!(lines, "")

    format_gauge(m.active_requests)
    push!(lines, "")

    format_gauge(m.model_load_time)

    return join(lines, "\n")
end

# =============================================================================
# Alerting
# =============================================================================

"""
Alert configuration.
"""
Base.@kwdef struct AlertConfig
    latency_p99_threshold::Float64 = 0.1     # 100ms
    error_rate_threshold::Float64 = 0.01      # 1%
    check_interval_seconds::Int = 60
end

"""
Alert status.
"""
struct AlertStatus
    timestamp::DateTime
    alerts::Vector{String}
    metrics_snapshot::Dict{String, Any}
end

"""
Check for alert conditions.
"""
function check_alerts(config::AlertConfig = AlertConfig())
    m = get_metrics()
    alerts = String[]

    # Calculate latency percentiles
    if m.request_latency.count > 0
        # Approximate p99 from histogram
        # This is a simplified calculation
        p99_bucket_idx = findfirst(
            i -> m.request_latency.bucket_counts[i] >= 0.99 * m.request_latency.count,
            1:length(m.request_latency.buckets)
        )

        if p99_bucket_idx !== nothing
            p99_approx = m.request_latency.buckets[p99_bucket_idx]
            if p99_approx > config.latency_p99_threshold
                push!(alerts, "High latency: P99 ≈ $(round(p99_approx * 1000, digits=1))ms > $(config.latency_p99_threshold * 1000)ms threshold")
            end
        end
    end

    # Calculate error rate
    total_requests = m.requests_total.value
    total_errors = m.errors_total.value

    if total_requests > 0
        error_rate = total_errors / total_requests
        if error_rate > config.error_rate_threshold
            push!(alerts, "High error rate: $(round(error_rate * 100, digits=2))% > $(config.error_rate_threshold * 100)% threshold")
        end
    end

    # Create snapshot
    snapshot = Dict{String, Any}(
        "total_requests" => total_requests,
        "total_errors" => total_errors,
        "active_requests" => m.active_requests.value,
        "avg_latency_ms" => m.request_latency.count > 0 ?
            round(m.request_latency.sum / m.request_latency.count * 1000, digits=2) : 0.0,
    )

    return AlertStatus(now(), alerts, snapshot)
end

# =============================================================================
# Statistics
# =============================================================================

"""
Get current statistics summary.
"""
function get_stats()
    m = get_metrics()

    uptime = now() - m.last_reset

    avg_latency = m.request_latency.count > 0 ?
        m.request_latency.sum / m.request_latency.count : 0.0

    avg_batch_size = m.batch_size.count > 0 ?
        m.batch_size.sum / m.batch_size.count : 0.0

    # Entity type distribution
    entity_distribution = Dict{String, Float64}()
    total_entities = sum(values(m.entities_total.labels))
    if total_entities > 0
        for (entity_type, count) in m.entities_total.labels
            entity_distribution[entity_type] = round(count / total_entities * 100, digits=1)
        end
    end

    return Dict(
        "uptime_seconds" => Dates.value(uptime) / 1000,
        "total_requests" => m.requests_total.value,
        "total_errors" => m.errors_total.value,
        "error_rate_percent" => m.requests_total.value > 0 ?
            round(m.errors_total.value / m.requests_total.value * 100, digits=2) : 0.0,
        "avg_latency_ms" => round(avg_latency * 1000, digits=2),
        "avg_batch_size" => round(avg_batch_size, digits=1),
        "tokens_processed" => m.tokens_processed.value,
        "entities_extracted" => sum(values(m.entities_total.labels)),
        "entity_distribution" => entity_distribution,
        "active_requests" => m.active_requests.value,
    )
end

"""
Print stats summary to console.
"""
function print_stats()
    stats = get_stats()

    println("\n" * "=" ^ 50)
    println("NER Service Statistics")
    println("=" ^ 50)

    println("Uptime: $(round(stats["uptime_seconds"] / 3600, digits=2)) hours")
    println("Total requests: $(Int(stats["total_requests"]))")
    println("Error rate: $(stats["error_rate_percent"])%")
    println("Avg latency: $(stats["avg_latency_ms"]) ms")
    println("Avg batch size: $(stats["avg_batch_size"])")
    println("Tokens processed: $(Int(stats["tokens_processed"]))")
    println("Entities extracted: $(Int(stats["entities_extracted"]))")

    if !isempty(stats["entity_distribution"])
        println("\nEntity distribution:")
        for (entity, pct) in sort(collect(stats["entity_distribution"]), by=x->-x[2])
            println("  $entity: $pct%")
        end
    end
end

# =============================================================================
# Exports
# =============================================================================

export Counter, Histogram, Gauge, MetricsRegistry
export get_metrics, reset_metrics!
export record_request!, record_error!, track_active_request, record_model_load_time!
export prometheus_format
export AlertConfig, AlertStatus, check_alerts
export get_stats, print_stats

end # module
