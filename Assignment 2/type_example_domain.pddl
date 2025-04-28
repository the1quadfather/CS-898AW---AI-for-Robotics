(define (domain <domain-name>)
  

(:requirements :strips :typing)

(:types
  robot location
)

(:action move
  :parameters (?r - robot ?from - location ?to - location)
  :precondition (and (at ?r ?from))
  :effect (and (not (at ?r ?from)) (at ?r ?to))
)

)
