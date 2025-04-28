(define (domain tool-affordance)
  (:requirements :strips :typing)

  (:types
    agent
    tool
    object
    affordance
    location
  )

  (:predicates
    ;; Agent possesses a tool
    (has ?a - agent ?t - tool)

    ;; Tool provides a specific affordance
    (affords ?t - tool ?f - affordance)

    ;; Agent is at a specific location
    (at ?a - agent ?l - location)

    ;; Affordance is used for a particular action on an object
    (used-for ?f - affordance ?act - object)

    ;; Object has been chopped
    (chopped ?o - object)

    ;; Object has been moored
    (moored ?o - object)

    ;; Object is being carried
    (carried ?o - object)
  )

  ;; Action: Use a tool to chop an object
  (:action use-axe
    :parameters (?a - agent ?t - tool ?o - object)
    :precondition (and
      (has ?a ?t)
      (affords ?t press)
    )
    :effect (chopped ?o)
  )

  ;; Action: Use a tool to moor an object
  (:action drop-anchor
    :parameters (?a - agent ?t - tool ?s - object)
    :precondition (and
      (has ?a ?t)
      (affords ?t support)
    )
    :effect (moored ?s)
  )

  ;; Action: Use a tool to carry an object
  (:action carry-bag
    :parameters (?a - agent ?t - tool ?i - object)
    :precondition (and
      (has ?a ?t)
      (affords ?t lift)
    )
    :effect (carried ?i)
  )
)
