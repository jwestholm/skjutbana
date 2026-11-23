"""Opt-in structural parts on the existing GameObjects, without detector/hardware."""
from types import SimpleNamespace as S
import unittest
from src.engine.game_objects import ObjectManager,ObjectGeometry,HitShapeSpec,make_breakable_object
from src.engine.game_objects.events import EffectAction,ReactionRule


class PartsTests(unittest.TestCase):
    def build(self,owner='a',entity='vase'):
        manager=ObjectManager()
        for part,y in [('body',100),('neck',60),('flower',20)]:
            obj=make_breakable_object(f'{owner}:{part}',ObjectGeometry(100,y,40,30),
                owner=owner,entity_id=entity,part_id=part,shape=HitShapeSpec.rect(),break_effect=part)
            children={'body':('neck','flower'),'neck':('flower',),'flower':()}[part]
            if children:obj.reactions.append(ReactionRule.on('object.broken',EffectAction.break_parts(*children),once=True))
            manager.add(obj)
        return manager
    def hit(self,m,part):
        o=m.get('a:'+part);x,y=o.geometry.center
        return m.resolve_hit(S(game_x=x,game_y=y,shot_id=None),active_only=True)
    def test_body_breaks_siblings_but_has_one_physical_interaction(self):
        m=self.build();r=self.hit(m,'body')
        self.assertEqual(len(r.interactions),1)
        self.assertTrue(all(not o.active and not o.visible and o.damage_model.depleted for o in m.objects))
        hits=[e for e in m.events.history if e.event_type=='shot.hit']
        self.assertEqual([e.object_id for e in hits],['a:body'])
        structural=[e for e in m.events.history if e.event_type=='object.broken' and e.payload.get('cause')=='structural']
        self.assertEqual({e.object_id for e in structural},{'a:neck','a:flower'})
        before=len(m.effect_requests);self.hit(m,'body');self.assertEqual(len(m.effect_requests),before)
    def test_neck_preserves_body_and_flower_preserves_everything_else(self):
        m=self.build();self.hit(m,'neck');self.assertTrue(m.get('a:body').active)
        self.assertFalse(m.get('a:flower').active)
        m=self.build();self.hit(m,'flower')
        self.assertTrue(m.get('a:body').active and m.get('a:neck').active)
    def test_scope_owner_and_entity(self):
        m=self.build()
        for other in [self.build('b','vase'),self.build('c','other')]:
            for o in other.objects:
                o.geometry.move(300,0);m.add(o)
        extra=make_breakable_object('other_neck',ObjectGeometry(200,20,20,20),owner='a',entity_id='other',part_id='neck')
        m.add(extra);self.hit(m,'body')
        self.assertTrue(all(o.active for o in m.objects if not o.object_id.startswith('a:')))
    def test_cyclic_links_are_bounded(self):
        m=self.build();m.get('a:flower').reactions.append(ReactionRule.on('object.broken',EffectAction.break_parts('body')))
        self.hit(m,'body');self.assertLess(len(m.events.history),40)
        self.assertEqual(len([e for e in m.events.history if e.event_type=='object.broken']),3)
    def test_no_opt_in_keeps_other_parts_and_bad_names_fail(self):
        m=self.build();m.get('a:body').reactions=m.get('a:body').reactions[:1]
        self.hit(m,'body');self.assertTrue(m.get('a:neck').active)
        with self.assertRaises(ValueError):EffectAction.break_parts('')
        with self.assertRaises(ValueError):EffectAction.break_parts()

if __name__=='__main__':unittest.main()
